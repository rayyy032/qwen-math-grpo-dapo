"""规则奖励与数值解析的边界测试。

覆盖 ``\\boxed{}`` 抽取（嵌套/末尾窗口/缺失）、``math_verify`` 数值等价
（分数、小数、千分位、科学计数法、单位文本）以及 Soft Overlong
Punishment 的分段边界。运行：``uv run pytest tests/ -q``。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reward import (  # noqa: E402
    answers_equivalent,
    extract_last_boxed,
    score_answer,
    soft_overlong_penalty,
)


# ---------------------------------------------------------------------------
# extract_last_boxed
# ---------------------------------------------------------------------------


def test_boxed_simple_integer() -> None:
    assert extract_last_boxed("blah \\boxed{42} done") == "42"


def test_boxed_nested_braces() -> None:
    assert extract_last_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_boxed_nested_deep() -> None:
    text = "\\boxed{\\sqrt{\\frac{3}{4}}}"
    assert extract_last_boxed(text) == "\\sqrt{\\frac{3}{4}}"


def test_boxed_takes_last_of_multiple() -> None:
    text = "\\boxed{1} middle \\boxed{2}"
    assert extract_last_boxed(text) == "2"


def test_boxed_missing_returns_none() -> None:
    assert extract_last_boxed("no answer here") is None


def test_boxed_unterminated_brace_ignored() -> None:
    # 最后一个 \boxed 缺右括号时，应回退到前一个完整的。
    text = "\\boxed{7} then \\boxed{broken"
    assert extract_last_boxed(text) == "7"


# ---------------------------------------------------------------------------
# answers_equivalent（math_verify 数值等价）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("42", "42"),
        ("42.0", "42"),
        ("1,000", "1000"),
        ("0.5", "\\frac{1}{2}"),
        ("\\frac{2}{4}", "\\frac{1}{2}"),
        ("-3", "-3"),
        ("-3", "(-3)"),
        ("1e3", "1000"),
        ("\\pi", "3.141592653589793"),
    ],
)
def test_answers_equivalent(prediction: str, reference: str) -> None:
    assert answers_equivalent(prediction, reference)


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("42", "43"),
        ("0.5", "\\frac{1}{3}"),
        ("-3", "3"),
        ("\\sqrt{2}", "1.5"),
    ],
)
def test_answers_not_equivalent(prediction: str, reference: str) -> None:
    assert not answers_equivalent(prediction, reference)


def test_answers_equivalent_garbage_is_safe() -> None:
    # math_verify 解析失败必须返回 False 而不是抛异常。
    assert answers_equivalent("", "42") is False


# ---------------------------------------------------------------------------
# soft_overlong_penalty 分段边界
# ---------------------------------------------------------------------------


def test_overlong_zero_before_cache() -> None:
    assert soft_overlong_penalty(500, max_tokens=1024, overlong_cache=256) == 0.0


def test_overlong_linear_inside_cache() -> None:
    # penalty 区间起点 1024-256=768；768+128 token 时惩罚 -(128/256)。
    value = soft_overlong_penalty(768 + 128, max_tokens=1024, overlong_cache=256)
    assert math.isclose(value, -0.5, abs_tol=1e-9)


def test_overlong_saturated_at_max_tokens() -> None:
    assert soft_overlong_penalty(1024, max_tokens=1024, overlong_cache=256) == -1.0


def test_overlong_beyond_max_tokens_stays_capped() -> None:
    assert soft_overlong_penalty(9999, max_tokens=1024, overlong_cache=256) == -1.0


def test_overlong_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        soft_overlong_penalty(10, max_tokens=100, overlong_cache=200)
    with pytest.raises(ValueError):
        soft_overlong_penalty(-1, max_tokens=100, overlong_cache=10)


# ---------------------------------------------------------------------------
# score_answer 端到端
# ---------------------------------------------------------------------------


def test_score_answer_correct_without_overlong() -> None:
    result = score_answer(
        "reasoning... \\boxed{72}",
        "72",
        completion_tokens=200,
        max_tokens=1024,
        overlong_cache=256,
        enable_overlong=False,
    )
    assert result.correct is True
    assert result.base_reward == 1.0
    assert result.length_penalty == 0.0
    assert result.shaped_reward == 1.0


def test_score_answer_wrong_answer_negative_reward() -> None:
    result = score_answer(
        "reasoning... \\boxed{71}",
        "72",
        completion_tokens=200,
        max_tokens=1024,
        overlong_cache=256,
        enable_overlong=False,
    )
    assert result.correct is False
    assert result.shaped_reward == -1.0


def test_score_answer_overlong_shaping() -> None:
    # 截断在 80% max_tokens 处：惩罚 -(819-768)/256。
    result = score_answer(
        "\\boxed{72}",
        "72",
        completion_tokens=819,
        max_tokens=1024,
        overlong_cache=256,
        enable_overlong=True,
    )
    assert result.correct is True
    assert math.isclose(result.length_penalty, -51 / 256, abs_tol=1e-9)
    assert result.shaped_reward == 1.0 + result.length_penalty


def test_score_answer_missing_boxed_counts_wrong() -> None:
    result = score_answer(
        "the answer is obviously 42",
        "42",
        completion_tokens=100,
        max_tokens=1024,
        overlong_cache=256,
        enable_overlong=False,
    )
    # 答案未放进 \boxed{}：格式不合法且判错（抑制 reward hacking 的
    # 策略之一——只在显式答案标记内匹配）。
    assert result.valid_format is False
    assert result.correct is False
