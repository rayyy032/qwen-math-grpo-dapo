# Qwen Math GRPO/DAPO · 从零手写的 GRPO 训练与 DAPO 消融

用纯 PyTorch + Transformers + PEFT 从零实现 GRPO 训练循环，在 Qwen2.5-0.5B-Instruct 上做数学推理 RL，并将 DAPO 的四项改进以**单变量开关**形式逐项消融，另加两个自选扩展：**熵正则**与 **reward-hacking 监控**。

不依赖 TRL / verl / OpenRLHF 等 RL 框架——组内相对优势、PPO-clip 目标、KL、LoRA 注入、left-padding 批量生成、response-span log-prob 切片，全部手写，每个张量的形状都可断点检查。

## 为什么需要 DAPO：GRPO 的三类失效模式

| 失效模式 | 现象 | DAPO 对策 | 本仓库开关 |
|---|---|---|---|
| 熵坍缩 | 训练后期策略确定性过强，`p_new/p_old` 全部贴住下界 1-ε，探索消失 | Clip-Higher：非对称 clip，上界放宽到 1+0.28 | `clip_higher` |
| 零方差组污染梯度 | 组内全对/全错 → advantage 全 0，batch 白跑、梯度噪声增大 | Dynamic Sampling：丢弃零方差组并补采 | `dynamic_sampling` |
| 长样本主导损失 | sample-mean 损失下，长回答被隐式加权，鼓励啰嗦 | Token-level Loss：全 batch 统一 token-mean | `token_level` |
| 截断样本的奖励悬崖 | 达到 max_new_tokens 直接硬截断，reward 突变刺伤策略 | Overlong Reward Shaping：分段线性软惩罚 | `overlong_shaping` |

## DAPO 四项：公式与实现

### 1. Clip-Higher（非对称裁剪）

$$
r_i(\theta)=\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{old}}(o_i\mid q)},\qquad
\hat A_i = \frac{r_i^{reward}-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)+\varepsilon}
$$

$$
\mathcal L_{policy}=-\mathbb E\Big[\min\big(r_i\hat A_i,\ \mathrm{clip}(r_i,\,1-\varepsilon_{low},\,1+\varepsilon_{high})\hat A_i\big)\Big]
$$

低概率 token（探索的来源）一旦被正 advantage 放大，很容易撞到对称上界 1+ε=1.2 被裁掉。DAPO 把上界放宽到 `1+eps_high=1.28`，下界保持 `1-eps_low=0.8`，只放宽"变强"的方向。实现见 `src/losses.py: grpo_loss(eps_low, eps_high)`。

### 2. Dynamic Sampling（动态采样补齐）

组内 reward 全相同时 `std=0`、advantage 全 0，这批样本不产生有效梯度。开启后每个 batch 丢弃零方差组并继续补采，直到凑满 `groups_per_step` 或达到 `dynamic_max_attempts` 倍上限。训练日志中的 `zero_var_ratio` / `effective_group_ratio` / `oversample_ratio` 即此机制的诊断量。实现见 `scripts/train.py: sample_groups`。

### 3. Token-level Loss（逐 token 损失）

GRPO 原始实现按样本均值再对 batch 均值，相当于给长回答隐式加权。改为全 batch 有效 response token 统一求均值：

$$
\mathcal L=\frac{1}{\sum_i|o_i|}\sum_i\sum_{t}\big[\min(\cdot)\big]_{i,t}
$$

实现见 `src/losses.py: grpo_loss(token_level=True)`。

### 4. Overlong Reward Shaping（超长软惩罚）

对达到长度预算被截断的回答，硬奖励（直接给 0/-1）会让 reward 出现悬崖。改为分段线性软惩罚（`src/rewards.py: soft_overlong_penalty`）：

$$
penalty(len)=
\begin{cases}
0, & len < L_{max}-L_{cache}\\[4pt]
-\dfrac{len-(L_{max}-L_{cache})}{L_{cache}}, & L_{max}-L_{cache}\le len < L_{max}\\[8pt]
-1, & len \ge L_{max}
\end{cases}
$$

进入软区后惩罚随长度线性加深，到预算上限饱和为 -1，不再有突变。

## 扩展 A：熵正则（探索保持）

熵坍缩只靠 Clip-Hier 缓解不够彻底，额外加显式熵奖励项，用采样 token 的平均 surprisal 近似策略熵：

$$
\mathcal L_{ent}=-\mathrm{ent\_coef}\cdot\frac{1}{|o|}\sum_t\big(-\log\pi_\theta(o_t\mid q,o_{<t})\big)
$$

对应第七组消融 `entropy_reg`（在 DAPO 全开基础上叠加），实现见 `src/losses.py: compute_surprisal`。

## 扩展 B：Reward-Hacking 监控

规则奖励（答案匹配 + 格式分）天然可被 hack。三个检测器**只记录指标、不进 reward**，用于在曲线异常前定位投机行为（`src/rewards.py: detect_reward_hacking`）：

- `length_exploit`：长回答得分显著高于短回答（啰嗦换分）
- `format_only_hack`：格式分占比异常升高（会写 `\boxed{}` 但答案错误）
- `equivalence_bypass`：等价判定通过但字符完全不符（等价逻辑漏洞）

## Reward 设计：math-verify 数值等价

答案判定三级（`src/rewards.py: answers_equivalent`）：

1. 精确字符串匹配
2. 纯数值 fast-path：剥离 `$`/`,`/空白 → `float` 解析 → `math.isclose`（覆盖 `1e3 == 1000`、`42.0 == 42`，绕过 math-verify 对科学计数法解析不全的问题）
3. math-verify 符号等价（`\frac{1}{2} == 0.5`）

`\boxed{}` 提取支持任意深度嵌套括号并取**最后一个**完整匹配（`extract_last_boxed`）。最终 reward = 答案对错（0/1）+ 格式分（`format_weight`）+ overlong 软惩罚。

## 工程细节：left padding 与 response span

decoder-only 模型批量生成必须 **left padding**（右侧 padding 会导致生成位置错乱）。本项目的批量流水线（`src/generation.py`）：

- 生成时 `padding_side="left"`，pad token 用 `<|endoftext|>`（与 eos 区分）
- 每行记录 response 的 `(start, end)` span（扫描 pad + eos 定位真实回答边界）
- log-prob 重算按 span 切片 gather，剔除 pad 位置的 logits 污染
- 训练全程零 "right-padding may cause unexpected behavior" 警告

配置上 `apply_preset` 自动钳制 `overlong_cache < max_new_tokens`，小预算冒烟不会触发非法参数。

## 七组单变量消融

| algorithm | clip_higher | dynamic_sampling | token_level | overlong_shaping | entropy_reg |
|---|:-:|:-:|:-:|:-:|:-:|
| `grpo`（基线） | | | | | |
| `clip_higher` | ✅ | | | | |
| `dynamic` | | ✅ | | | |
| `token_level` | | | ✅ | | |
| `overlong` | | | | ✅ | |
| `dapo`（四项全开） | ✅ | ✅ | ✅ | ✅ | |
| `entropy_reg`（DAPO+熵） | ✅ | ✅ | ✅ | ✅ | ✅ |

七组除算法开关外**全部超参一致**（preset 表见 `src/config.py`），保证差异只来自算法本身。

## 快速开始

```bash
# 安装
pip install -r requirements.txt

# 27 项边界单测（boxed 提取 / 数值等价 / overlong 边界 / 端到端打分）
python -m pytest tests/ -q

# CPU 冒烟（合成数据，0.5B + LoRA，验证整条训练流水线）
python scripts/train.py --algorithm dapo --dataset synthetic \
    --train-samples 20 --max-steps 2 --device cpu

# 正式训练（GSM8K，单卡）
python scripts/train.py --algorithm entropy_reg --dataset gsm8k \
    --train-samples 500 --max-steps 100

# 七组消融一键跑（BASE_MODEL / DATASET / MAX_STEPS 可用环境变量覆盖）
bash scripts/run_ablations.sh
```

训练曲线与诊断面板（reward / accuracy / surprisal / clip_fraction / 零方差组占比 / hack rates）默认落盘到 `outputs/` 与 `swanlog/`（SwanLab 本地模式，无需注册账号）。

## 项目结构

```
src/
  losses.py       # GRPO 目标：非对称 clip / token-level / KL / 熵正则 / 诊断统计
  rewards.py      # boxed 提取、三级数值等价、overlong 软惩罚、hack 检测
  generation.py   # left-padding 批量生成 + response-span log-prob 切片
  config.py       # dataclass 配置 + 七组 preset + CLI 覆盖
  data.py         # GSM8K / MATH 分层数据构建 + 离线合成集
  evaluation.py   # 贪心解码评测（accuracy / format / 分来源）
  diagnostics.py  # 训练指标聚合
  agents/         # Qwen + LoRA 策略封装、frozen reference
scripts/
  train.py        # 统一训练入口（dynamic sampling 主循环）
  run_ablations.sh
tests/
  test_reward.py  # 27 项边界单测
```

## Acknowledgments

- 训练循环的结构参考 [Feld-maxiu/bachelor-thesis-rl](https://github.com/Feld-maxiu/bachelor-thesis-rl)（from-scratch GRPO 骨架），算法扩展与实现为本仓库工作。
- DAPO 算法来自 [BytedTsinghua-SIA/DAPO](https://github.com/bytedance/DAPO) 论文 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*。

## License

Apache-2.0
