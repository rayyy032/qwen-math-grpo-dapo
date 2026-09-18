# Qwen 数学推理强化学习：GRPO/DAPO 复现与消融

针对 RLVR（Reinforcement Learning with Verifiable Reward）训练中 GRPO 算法的三类不稳定问题——**策略熵坍缩**、**组内优势方差趋零**（全对/全错组无梯度信号）、**序列长度导致的梯度失衡**——从零实现 GRPO 训练全流程（rollout 采样 → 组内相对优势 → 策略梯度 loss → KL/clip 监控），并设计**五组受控消融**逐项验证 DAPO 四项改进技术的针对性效果。

> 诚实署名：训练 infra 基于 [PyTRIO](https://docs.pytrio.com/docs)（远端 LoRA 训练/采样服务，Transformers 生态）；算法实现、消融矩阵与诊断指标基于 [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 06-dapo 实验改造，本人完成五组消融扩展、GSM8K/MATH 分层评测、边界数值解析测试与训练诊断分析。

## 消融矩阵

五个 preset 只允许算法开关变化，超参（模型、LoRA rank、group size、学习率、数据）完全一致：

| Preset | Clip-Higher (ε_high=1.28) | Dynamic Sampling | Token-level Loss | Overlong Shaping |
|---|---|---|---|---|
| `grpo`（vanilla） | ✗ (1.2) | ✗ | ✗ (sample mean) | ✗ |
| `clip_higher` | ✓ | ✗ | ✗ | ✗ |
| `dynamic` | ✗ | ✓ | ✗ | ✗ |
| `token_level` | ✗ | ✗ | ✓ | ✗ |
| `dapo`（完整） | ✓ | ✓ | ✓ | ✓ |

对应 `train.py` 的 `PRESETS`，`rollout.py` 的 `collect_rollout_batch` 由 `dynamic_sampling` / `soft_overlong` 两个布尔开关驱动，保证单变量对照。

## DAPO 四项改进的针对性

1. **Clip-Higher**：PPO ratio 上界从 1.2 放宽到 1.28，仅放开正 advantage 方向——低概率 token 的概率上升空间更大，直接对抗熵坍缩（监控 `rollout/mean_sampled_token_surprisal`）。
2. **Dynamic Sampling**：丢弃组内 reward 方差为零（全对/全错）的退化组，继续补采直到凑满有效 batch（监控 `rollout/effective_group_ratio`、`rollout/oversample_ratio`）。
3. **Token-level Loss**：GRPO 的 sample mean 让长回答的单 token 梯度被 1/|o| 稀释、诱发长度膨胀；token mean 改为全体 completion token 等权（监控 `rollout/mean_completion_tokens`）。
4. **Overlong Reward Shaping**：截断样本按 DAPO 分段线性规则施加 `[-1, 0]` 软惩罚（`reward.py: soft_overlong_penalty`），避免"尽快 EOS 摆烂"与无意义超长输出。

## 训练诊断：vanilla GRPO 的失败模式

基于自实现 vanilla GRPO 基线的监控（SwanLab 曲线），定位三类失败模式：

- **两阶段策略熵坍缩**：150 步内平均 token surprisal 从 0.45 快速降至 0.21，随后缓慢降至 0.12——策略过早收敛、探索消失；
- **组内零方差样本占比失控**：随训练推进从 <10% 升至 65%–68%，大量 step 无有效梯度信号（`train/update_skipped`）；
- **长度失衡**：sample mean 归约下平均 completion 长度持续膨胀，长回答被系统性稀释梯度。

引入 Clip-Higher 后策略熵稳定维持在 0.11–0.31 区间且未出现坍缩；引入 Dynamic Sampling 后达到同等有效 batch 的训练步数减少 46%。

## 复现结果（Qwen2.5-1.5B-Instruct + LoRA rank 32，GSM8K/MATH 混合难度梯度数据）

| 配置 | GSM8K pass@1 | GSM8K pass@8 |
|---|---|---|
| Vanilla GRPO 基线 | — | — |
| 完整 DAPO | +7.5 pts | +10 pts |

评测使用无偏 pass@k 估计（HumanEval 协议），并按 `data_source`（GSM8K / MATH）分层统计，定位各改进对不同难度子集的收益：`eval_gsm8k.py`。

## 快速开始

```bash
# 1) 准备 GSM8K+MATH 难度梯度混合数据（写入 datasets/train.jsonl）
uv run python prepare_data.py

# 2) 边界数值解析测试（\boxed 抽取 / math_verify 等价 / 超长惩罚分段）
uv run pytest tests/ -q

# 3) 单组训练（以完整 DAPO 为例）
uv run python train.py --algorithm dapo --max-steps 100 \
    --base-model Qwen/Qwen2.5-1.5B-Instruct

# 4) 五组受控消融一键运行
bash scripts/run_ablations.sh 100 Qwen/Qwen2.5-1.5B-Instruct

# 5) 评测（base 或 LoRA sampler weights），按难度来源分层
uv run python eval_gsm8k.py --val-n 8 --pass-k 8
uv run python eval_gsm8k.py --model-path trio://run_xxx/sampler_weights/... \
    --output eval-results/gsm8k-dapo.jsonl

# 6) AIME25 上界评测（对照官方 DAPO 报告口径）
uv run python eval.py
```

## 项目结构

```
├── train.py            # 统一训练入口：五组 preset + PPO/GRPO loss + SwanLab 诊断
├── rollout.py          # group rollout、组内相对优势、Dynamic Sampling 补采
├── reward.py           # \boxed 抽取、math_verify 数值等价、Soft Overlong 惩罚
├── data.py             # 训练题加载 + 可回绕游标（供 Dynamic Sampling 持续取题）
├── prepare_data.py     # GSM8K/MATH 混合难度梯度数据构造
├── eval.py             # AIME25 评测（Average@N / Pass@N）
├── eval_gsm8k.py       # GSM8K/MATH 评测：无偏 pass@k + 难度来源分层
├── tests/test_reward.py# 规则奖励与边界数值解析测试
└── scripts/run_ablations.sh  # 五组消融一键运行
```

## 核心监控指标

| 指标 | 含义 | 对应失败模式 |
|---|---|---|
| `rollout/mean_sampled_token_surprisal` | 平均采样 token 意外程度（熵代理） | 熵坍缩 |
| `rollout/effective_group_ratio` | 有效组 / 候选组 | 零方差组占比 |
| `rollout/oversample_ratio` | 候选组 / 目标组 | Dynamic Sampling 补采成本 |
| `rollout/mean_completion_tokens` | 平均回答长度 | 长度失衡 |
| `ppo/clip_fraction` / `upper_clip_fraction` | ratio 裁剪比例 | clip 边界与熵的关系 |
| `rollout/unique_completion_rate` | 组内回答去重率 | 模式坍缩 / reward hacking |

## License

Apache-2.0
