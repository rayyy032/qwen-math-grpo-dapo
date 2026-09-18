"""在 GSM8K / MATH 测试集上评测 base model 或 PyTRIO LoRA sampler weights。

与 ``eval.py``（AIME25）共用采样骨架，本脚本面向消融对照：

- Average@k（每题 k 条采样的平均正确率）与无偏 Pass@k 估计；
- 按 ``data_source`` 分层统计（GSM8K / MATH 难度梯度），用于定位
  各算法开关对不同难度子集的针对性收益。

Base model：

    uv run python eval_gsm8k.py

LoRA：

    uv run python eval_gsm8k.py \
        --model-path trio://run_xxx/sampler_weights/weights-name \
        --output eval-results/gsm8k-dapo-100steps.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

import pytrio as trio
from tqdm import tqdm

from data import load_examples
from reward import answers_equivalent, extract_last_boxed
from rollout import build_prompt_tokens, stop_sequences


trio.configure(sampling_timeout=18000)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "test.jsonl"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "eval-results" / "gsm8k-base.jsonl"


def pass_at_k(n_correct: int, n_total: int, k: int) -> float:
    """无偏 pass@k 估计（HumanEval 协议）。"""
    if n_correct >= n_total:
        return 1.0
    if k > n_total:
        k = n_total
    return 1.0 - comb(n_total - n_correct, k) / comb(n_total, k)


def parse_args() -> argparse.Namespace:
    """解析 GSM8K/MATH 评测参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--model-path",
        default=None,
        help="可选；save_weights_for_sampler 返回的 trio:// 路径，不传则评测 base model",
    )
    parser.add_argument("--val-n", type=int, default=8)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="结果 JSONL 路径；base 默认写入 gsm8k-base.jsonl，LoRA 必须手动指定",
    )
    args = parser.parse_args()

    for name in ("val_n", "pass_k", "concurrency", "max_prompt_tokens", "max_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.pass_k > args.val_n:
        raise ValueError("--pass-k must not exceed --val-n")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid sampling temperature/top-p")
    if args.model_path is not None and args.output is None:
        raise ValueError("--model-path requires --output")
    args.output = args.output or DEFAULT_OUTPUT_PATH
    return args


async def evaluate_problem(
    index: int,
    row: dict[str, Any],
    sampling_client: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """并发采样一道题的 N 个答案并判定正确性。"""
    question = str(row["question"]).strip()
    reference = str(row["answer"]).strip()
    prompt_tokens = build_prompt_tokens(
        tokenizer,
        question,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed + index,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=stop_sequences(tokenizer),
    )
    async with semaphore:
        response = await sampling_client.sample_async(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
            num_samples=args.val_n,
            sampling_params=params,
            return_text=True,
        )

    generations: list[dict[str, Any]] = []
    for sequence in response.sequences:
        text = sequence.text
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        boxed = extract_last_boxed(str(text))
        correct = boxed is not None and answers_equivalent(boxed, reference)
        generations.append(
            {
                "predicted_answer": boxed,
                "correct": correct,
                "formatted": boxed is not None,
                "completion_tokens": len(sequence.tokens),
                "stop_reason": str(getattr(sequence, "stop_reason", "")),
                "text": str(text),
            }
        )
    n_correct = sum(int(item["correct"]) for item in generations)
    return {
        "type": "problem",
        "problem_id": str(row.get("id", index)),
        "data_source": str(row.get("data_source", "unknown")),
        "ground_truth": reference,
        "val_n": args.val_n,
        "num_correct": n_correct,
        "pass_at_k": pass_at_k(n_correct, args.val_n, args.pass_k),
        "generations": generations,
    }


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """整体 + 按 data_source 分层的 Average@N 与 Pass@k。"""
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {}
        generations = [g for r in rows for g in r["generations"]]
        total = len(generations)
        correct = sum(g["correct"] for g in generations)
        return {
            "problems": float(len(rows)),
            "average_at_n": correct / total if total else 0.0,
            "pass_at_k": (
                sum(r["pass_at_k"] for r in rows) / len(rows) if rows else 0.0
            ),
            "format_rate": (
                sum(g["formatted"] for g in generations) / total if total else 0.0
            ),
            "mean_completion_tokens": (
                sum(g["completion_tokens"] for g in generations) / total if total else 0.0
            ),
        }

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_source[result["data_source"]].append(result)

    return {
        "type": "summary",
        "dataset": str(args.dataset_path),
        "base_model": args.base_model,
        "model_path": args.model_path,
        "val_n": args.val_n,
        "pass_k": args.pass_k,
        "overall": aggregate(results),
        "by_data_source": {
            source: aggregate(rows) for source, rows in sorted(by_source.items())
        },
    }


async def evaluate(args: argparse.Namespace) -> None:
    """创建采样 client，完成评测并按难度来源分层落盘。"""
    examples = load_examples(args.dataset_path)
    if args.limit > 0:
        examples = examples[: args.limit]
    rows = [
        {
            "id": e.id,
            "question": e.question,
            "answer": e.answer,
            "data_source": e.data_source,
        }
        for e in examples
    ]

    service_client = trio.ServiceClient()
    client_kwargs = {"base_model": args.base_model}
    if args.model_path is not None:
        client_kwargs["model_path"] = args.model_path
    sampling_client = await service_client.create_sampling_client_async(**client_kwargs)
    tokenizer = sampling_client.get_tokenizer()
    semaphore = asyncio.Semaphore(args.concurrency)

    with tqdm(total=len(rows), desc="GSM8K/MATH", unit="problem") as progress:

        async def evaluate_and_track(index: int, row: dict[str, Any]) -> dict[str, Any]:
            result = await evaluate_problem(
                index, row, sampling_client, tokenizer, args, semaphore
            )
            progress.update(1)
            return result

        results = list(
            await asyncio.gather(
                *(evaluate_and_track(i, row) for i, row in enumerate(rows))
            )
        )

    summary = summarize(results, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    model_type = "base" if args.model_path is None else "LoRA"
    print(f"[{model_type}] Average@{args.val_n}: {summary['overall']['average_at_n']:.2%} | "
          f"Pass@{args.pass_k}: {summary['overall']['pass_at_k']:.2%}")
    for source, metrics in summary["by_data_source"].items():
        print(f"  - {source}: Avg@{args.val_n} {metrics['average_at_n']:.2%} | "
              f"Pass@{args.pass_k} {metrics['pass_at_k']:.2%}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
