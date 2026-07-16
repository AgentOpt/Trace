import copy
import warnings
from typing import Any, Dict, Optional, Tuple

import pytest

from opto.trainer.guide import Guide, TokenUsageAugmentingGuide, UsageTrackingLLM


class _Usage:
    def __init__(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str = "ok", usage: Optional[_Usage] = None) -> None:
        self.choices = [_Choice(content)]
        if usage is not None:
            self.usage = usage


class _BaseGuide(Guide):
    def get_feedback(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[float, str]:
        return 1.0, "base-feedback"

    def get_score_dict(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {"error": 0.0}


class _TokenCollisionGuide(_BaseGuide):
    def get_score_dict(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {"error": 0.0, "tokens_in": 1.0}


def test_usage_tracking_reads_openai_style_usage() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: _Response(
            content="two words",
            usage=_Usage(prompt_tokens=7, completion_tokens=2),
        ),
        estimate_missing=False,
    )

    tracker(messages=[{"role": "user", "content": "ignored"}])

    assert tracker.last_usage() == {"tokens_in": 7, "tokens_out": 2}
    assert tracker.has_usage()
    assert tracker.last_usage_was_estimated() is False


def test_usage_tracking_accepts_dict_usage_aliases() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: {
            "choices": [{"message": {"content": "answer text"}}],
            "usage": {"input_tokens": 9, "output_tokens": 3},
        },
        estimate_missing=False,
    )

    tracker(messages=[{"role": "user", "content": "ignored"}])

    assert tracker.last_usage() == {"tokens_in": 9, "tokens_out": 3}


def test_usage_tracking_estimates_missing_usage_when_allowed() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: _Response(content="two words"),
        estimate_missing=True,
    )

    with pytest.warns(UserWarning, match="estimating token counts"):
        tracker(messages=[{"role": "user", "content": "one two three"}])

    assert tracker.last_usage() == {"tokens_in": 3, "tokens_out": 2}
    assert tracker.last_usage_was_estimated() is True

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tracker(messages=[{"role": "user", "content": "one two three"}])

    assert captured == []


def test_usage_tracking_strict_mode_rejects_missing_usage() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: _Response(content="two words"),
        estimate_missing=False,
    )

    with pytest.raises(ValueError, match="did not include complete token usage"):
        tracker(messages=[{"role": "user", "content": "one two three"}])


def test_token_usage_augmenting_guide_requires_tracker_interface() -> None:
    with pytest.raises(TypeError, match="last_usage"):
        TokenUsageAugmentingGuide(_BaseGuide(), object())


def test_token_usage_augmenting_guide_adds_metrics_and_resets() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: _Response(
            content="ok",
            usage=_Usage(prompt_tokens=5, completion_tokens=1),
        )
    )
    guide = TokenUsageAugmentingGuide(_BaseGuide(), tracker)

    tracker(messages=[{"role": "user", "content": "ignored"}])
    score_dict = guide.get_score_dict("q", "r", "ref")

    assert score_dict == {"error": 0.0, "tokens_in": 5.0, "tokens_out": 1.0}
    assert tracker.has_usage() is False


def test_token_usage_augmenting_guide_feedback_includes_metrics() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: _Response(
            content="ok",
            usage=_Usage(prompt_tokens=4, completion_tokens=2),
        )
    )
    guide = TokenUsageAugmentingGuide(_BaseGuide(), tracker)

    tracker(messages=[{"role": "user", "content": "ignored"}])
    reward, feedback = guide.get_feedback("q", "r", "ref")

    assert reward == 1.0
    assert "base-feedback" in feedback
    assert "tokens_in=4" in feedback
    assert "tokens_out=2" in feedback


def test_token_usage_augmenting_guide_fails_when_no_usage_was_recorded() -> None:
    tracker = UsageTrackingLLM(lambda **kwargs: _Response())
    guide = TokenUsageAugmentingGuide(_BaseGuide(), tracker)

    with pytest.raises(RuntimeError, match="No token usage was recorded"):
        guide.get_score_dict("q", "r", "ref")


def test_token_usage_augmenting_guide_rejects_metric_collisions() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: _Response(
            content="ok",
            usage=_Usage(prompt_tokens=5, completion_tokens=1),
        )
    )
    guide = TokenUsageAugmentingGuide(_TokenCollisionGuide(), tracker)

    tracker(messages=[{"role": "user", "content": "ignored"}])

    with pytest.raises(ValueError, match="already emitted token metric keys"):
        guide.get_score_dict("q", "r", "ref")


def test_token_usage_survives_independent_agent_and_guide_deepcopy() -> None:
    tracker = UsageTrackingLLM(
        lambda **kwargs: _Response(
            content="ok",
            usage=_Usage(prompt_tokens=11, completion_tokens=4),
        )
    )

    class _Agent:
        def __init__(self, llm: UsageTrackingLLM) -> None:
            self.llm = llm

        def __call__(self, x: str) -> _Response:
            return self.llm(messages=[{"role": "user", "content": x}])

    agent = _Agent(tracker)
    guide = TokenUsageAugmentingGuide(_BaseGuide(), tracker)

    agent_copy = copy.deepcopy(agent)
    guide_copy = copy.deepcopy(guide)

    agent_copy("question")
    score_dict = guide_copy.get_score_dict("q", "r", "ref")

    assert score_dict["tokens_in"] == 11.0
    assert score_dict["tokens_out"] == 4.0
