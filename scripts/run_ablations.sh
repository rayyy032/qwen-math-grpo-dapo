#!/usr/bin/env bash
# 五组受控消融：Vanilla GRPO / +Clip-Higher / +Dynamic Sampling /
# +Token-level Loss / 完整 DAPO，统一超参依次训练并评测。
#
# 用法：
#   bash scripts/run_ablations.sh [max_steps] [base_model]
#
# 所有 run 仅算法 preset 不同，其余超参完全一致；训练完成后依次在
# GSM8K/MATH 测试集上评测 pass@k 并按 data_source 分层。

set -euo pipefail

MAX_STEPS="${1:-100}"
BASE_MODEL="${2:-Qwen/Qwen2.5-1.5B-Instruct}"
GROUPS=4
GROUP_SIZE=8

ALGORITHMS=(grpo clip_higher dynamic token_level dapo)

for algo in "${ALGORITHMS[@]}"; do
    echo "=== [ablation] training ${algo} for ${MAX_STEPS} steps ==="
    uv run python train.py \
        --algorithm "${algo}" \
        --max-steps "${MAX_STEPS}" \
        --base-model "${BASE_MODEL}" \
        --groups-per-step "${GROUPS}" \
        --group-size "${GROUP_SIZE}" \
        --max-prompt-tokens 512 \
        --max-tokens 4096 \
        --overlong-cache 1024 \
        --run-name "ablation-${algo}-${MAX_STEPS}steps"

    echo "=== [ablation] evaluating ${algo} ==="
    # 逐 run 手动填入 swanlog/终端打印的 sampler weights trio:// 路径：
    # uv run python eval_gsm8k.py \
    #     --model-path "trio://run_xxx/sampler_weights/ablation-${algo}-..." \
    #     --output "eval-results/ablation-${algo}.jsonl"
done

echo "=== [ablation] all training runs finished ==="
