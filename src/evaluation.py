"""Greedy-decoding evaluation with per-source (dataset) breakdown."""

from __future__ import annotations

from collections import defaultdict

import torch

from src.rewards import score_answer


@torch.no_grad()
def evaluate_model(agent, eval_data, max_new_tokens=256, overlong_cache=128,
                   enable_overlong=False, format_weight=0.2):
    """Evaluate accuracy / format rate on ``eval_data`` (near-greedy decode).

    Returns a dict with ``accuracy``, ``format_rate`` and ``per_source``
    (accuracy grouped by each sample's ``data_source`` field).
    """
    agent.policy_model.eval()
    n = len(eval_data)
    correct = 0
    valid_format = 0
    per_source = defaultdict(lambda: [0, 0])  # source -> [correct, total]

    for sample in eval_data:
        prompt = sample["prompt"]
        gt = sample["ground_truth"]
        source = sample.get("data_source", "unknown")
        # near-greedy: temperature ~0 under do_sample=True
        responses, _full_ids, _spans, comp_lens, truncated = \
            agent.generate_responses(
                [prompt], max_new_tokens=max_new_tokens, temperature=0.01,
                max_prompt_tokens=512)
        r = score_answer(responses[0], gt, comp_lens[0], max_new_tokens,
                         overlong_cache, enable_overlong, truncated[0],
                         format_weight)
        correct += int(r.correct)
        valid_format += int(r.valid_format)
        per_source[source][0] += int(r.correct)
        per_source[source][1] += 1

    per_source_out = {k: round(v[0] / v[1], 4) for k, v in per_source.items()}
    return {
        "accuracy": round(correct / max(n, 1), 4),
        "format_rate": round(valid_format / max(n, 1), 4),
        "per_source": per_source_out,
    }
