# 快速启动

以下命令均从仓库根目录运行。项目要求 Python `>=3.13`。

```bash
uv sync
trio login
swanlab login
```

## 1. 下载数据

```bash
uv run python prepare_data.py
```

这条命令会准备 GSM8K/MATH 混合训练集和 AIME25 评测集。

## 2. 边界数值解析测试

```bash
uv run pytest tests/ -q
```

覆盖 `\boxed{}` 嵌套抽取、math_verify 数值等价与 Soft Overlong 惩罚的分段边界。

## 3. 启动训练

```bash
uv run python train.py \
    --algorithm dapo \
    --max-steps 10 \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --groups-per-step 4 \
    --group-size 8 \
    --max-candidate-multiplier 2 \
    --max-prompt-tokens 512 \
    --max-tokens 4096 \
    --overlong-cache 1024 \
    --swanlab-mode online
```

同一入口支持五组消融：`--algorithm grpo | clip_higher | dynamic | token_level | dapo`，除算法开关外超参完全一致（matched 设置）。

## 4. 运行评测

GSM8K/MATH（pass@k + 难度来源分层），先评 Base Model：

```bash
uv run python eval_gsm8k.py
```

再评测训练得到的 sampler weights：

```bash
uv run python eval_gsm8k.py \
    --model-path 'trio://RUN_ID/sampler_weights/WEIGHTS_NAME' \
    --output eval-results/gsm8k-dapo-step10.jsonl
```

AIME25 上界对照：

```bash
uv run python eval.py
```
