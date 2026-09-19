"""Configuration: dataclass + algorithm presets + argparse merge.

The whole project is driven by a single :class:`Config` dataclass.  The
:data:`PRESETS` table maps an ``--algorithm`` name to the four DAPO toggles
plus the entropy-regularization extension, so an ablation grid keeps every
hyper-parameter identical except for the algorithm switches.
"""

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import Optional

# Make sure HF downloads happen through the mirror even before transformers
# is imported anywhere else.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ---------------------------------------------------------------------------
# Preset table (single-variable ablation: only toggles differ)
# ---------------------------------------------------------------------------
# clip_higher / dynamic_sampling / token_level / overlong_shaping / entropy_reg
PRESETS = {
    "grpo":        dict(clip_higher=False, dynamic_sampling=False, token_level=False,
                        overlong_shaping=False, entropy_reg=False),
    "clip_higher": dict(clip_higher=True,  dynamic_sampling=False, token_level=False,
                        overlong_shaping=False, entropy_reg=False),
    "dynamic":     dict(clip_higher=False, dynamic_sampling=True,  token_level=False,
                        overlong_shaping=False, entropy_reg=False),
    "token_level": dict(clip_higher=False, dynamic_sampling=False, token_level=True,
                        overlong_shaping=False, entropy_reg=False),
    "overlong":    dict(clip_higher=False, dynamic_sampling=False, token_level=False,
                        overlong_shaping=True,  entropy_reg=False),
    "dapo":        dict(clip_higher=True,  dynamic_sampling=True,  token_level=True,
                        overlong_shaping=True,  entropy_reg=False),
    "entropy_reg": dict(clip_higher=True,  dynamic_sampling=True,  token_level=True,
                        overlong_shaping=True,  entropy_reg=True),
}


@dataclass
class Config:
    # ---- model & device ----------------------------------------------------
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"                     # auto | cuda | mps | cpu
    use_8bit: bool = False                   # bitsandbytes 8-bit (CUDA only)

    # ---- LoRA --------------------------------------------------------------
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = field(default=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ))
    ref_use_lora: bool = False               # reference frozen base (no LoRA) by default

    # ---- algorithm (resolved from preset) ----------------------------------
    algorithm: str = "grpo"
    clip_higher: bool = False
    dynamic_sampling: bool = False
    token_level: bool = False
    overlong_shaping: bool = False
    entropy_reg: bool = False

    # ---- PPO / float / KL --------------------------------------------------
    epsilon: float = 0.2                     # base clip range = eps_low
    eps_high: float = 0.28                   # widened upper bound for clip-higher
    eps_low: float = 0.2
    beta: float = 0.0                        # KL penalty coefficient
    entropy_coef: float = 0.01               # entropy regularization strength

    # ---- training ----------------------------------------------------------
    seed: int = 42
    learning_rate: float = 1e-5
    ppo_epochs: int = 1
    max_grad_norm: float = 1.0
    group_size: int = 8
    groups_per_step: int = 2
    max_new_tokens: int = 256                # response generation budget
    max_prompt_tokens: int = 512             # prompt truncation limit
    temperature: float = 1.0
    max_steps: int = 100
    dynamic_max_attempts: int = 4            # max_candidate_multiplier for dynamic sampling

    # ---- reward ------------------------------------------------------------
    overlong_cache: int = 128                # soft zone width before response budget
    format_weight: float = 0.2

    # ---- data --------------------------------------------------------------
    dataset: str = "synthetic"               # synthetic | gsm8k | gsm8k_math | /path/to.jsonl
    train_samples: int = 500
    eval_samples: int = 100

    # ---- eval / logging / checkpoint --------------------------------------
    eval_every: int = 10
    save_every: int = 50
    use_swanlab: bool = True
    swanlab_project: str = "qwen-math-grpo-dapo"
    run_name: str = ""
    output_dir: str = "./outputs"
    log_dir: str = "./outputs/runs"
    resume_from: Optional[str] = None


def apply_preset(cfg: Config) -> Config:
    """Resolve the ``--algorithm`` name into DAPO toggles + adaptive clip range."""
    preset = PRESETS.get(cfg.algorithm, PRESETS["grpo"])
    for key, value in preset.items():
        setattr(cfg, key, value)

    # Clip-Higher only changes eps_high; vanilla keeps a symmetric clip.
    cfg.eps_low = cfg.epsilon
    cfg.eps_high = 0.28 if cfg.clip_higher else cfg.epsilon
    # keep the soft-penalty window valid for small generation budgets (smoke)
    if cfg.overlong_cache >= cfg.max_new_tokens:
        cfg.overlong_cache = max(1, cfg.max_new_tokens // 4)
    if cfg.run_name == "":
        cfg.run_name = cfg.algorithm
    return cfg


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def _type_of(field_obj, raw_value: str):
    """Coerce a CLI string into the declared dataclass field type."""
    t = field_obj.type
    if t is bool or t == "bool":
        return raw_value.lower() in ("true", "1", "yes")
    if t is int or t == "int":
        return int(raw_value)
    if t is float or t == "float":
        return float(raw_value)
    if t is Optional[int]:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return None
    return raw_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRPO / DAPO from-scratch reproduction + ablation grid.")
    parser.add_argument("--algorithm", type=str, default="grpo",
                        choices=list(PRESETS.keys()))
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--groups-per-step", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--use-swanlab", type=lambda s: s.lower() in ("true", "1", "yes"),
                        default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    # Generic --key value overrides for any remaining Config field.
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser


def load_config(argv=None) -> Config:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    cfg = Config()

    # explicit flags first
    for name in ("base_model", "device", "max_steps", "group_size", "groups_per_step",
                 "max_new_tokens", "max_prompt_tokens", "dataset", "train_samples",
                 "use_swanlab", "run_name", "output_dir"):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
    if args.algorithm:
        cfg.algorithm = args.algorithm

    # generic --key value overrides
    known = {f.name for f in fields(Config)}
    i = 0
    while i < len(extras):
        arg = extras[i]
        if arg.startswith("--") and i + 1 < len(extras):
            key = arg[2:].replace("-", "_")
            if key in known:
                fld = fields(Config)
                for f in fld:
                    if f.name == key:
                        setattr(cfg, key, _type_of(f, extras[i + 1]))
                        break
                i += 2
                continue
        i += 1

    return apply_preset(cfg)
