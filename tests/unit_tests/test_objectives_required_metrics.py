from typing import Dict

import pytest

from opto.trainer.objectives import ObjectiveConfig, select_best, select_top_k


def _token_objective_config() -> ObjectiveConfig:
    """Return a reusable objective that minimizes error and token usage."""
    return ObjectiveConfig(
        mode="weighted",
        weights={"error": 1.0, "tokens_in": 1e-3, "tokens_out": 1e-3},
        minimize=frozenset({"error", "tokens_in", "tokens_out"}),
        required_metrics=frozenset({"error", "tokens_in", "tokens_out"}),
    )


def test_weighted_objective_can_minimize_token_metrics() -> None:
    config = _token_objective_config()
    candidates = [
        ({"error": 0.0, "tokens_in": 100.0, "tokens_out": 100.0}, "long"),
        ({"error": 0.0, "tokens_in": 10.0, "tokens_out": 10.0}, "short"),
    ]

    assert select_best(candidates, config) == 1


def test_top_k_validates_required_metrics() -> None:
    config = _token_objective_config()
    candidates = [
        ({"error": 0.0, "tokens_in": 10.0, "tokens_out": 10.0}, "ok"),
        ({"error": 0.0, "tokens_in": 10.0}, "missing-tokens-out"),
    ]

    with pytest.raises(ValueError, match="Missing required objective metrics"):
        select_top_k(candidates, config, k=1)


def test_required_metrics_rejects_missing_token_metrics() -> None:
    config = _token_objective_config()
    candidates = [
        ({"error": 0.0, "tokens_in": 10.0}, "missing-tokens-out"),
    ]

    with pytest.raises(ValueError, match="Missing required objective metrics"):
        select_best(candidates, config)


def test_required_metrics_rejects_scalar_scores() -> None:
    config = _token_objective_config()

    with pytest.raises(ValueError, match="requires dict scores"):
        select_best([(1.0, "scalar-score")], config)


def test_required_metrics_accepts_sets_and_rejects_bad_names() -> None:
    config = ObjectiveConfig(required_metrics={"score"})

    assert config.required_metrics == frozenset({"score"})

    with pytest.raises(ValueError, match="required_metrics"):
        ObjectiveConfig(required_metrics={""})


def test_required_metrics_with_scalar_mode_validates_dict_scores() -> None:
    config = ObjectiveConfig(
        mode="scalar",
        scalarize_dict="score",
        required_metrics=frozenset({"score", "tokens_in"}),
    )
    score: Dict[str, float] = {"score": 1.0}

    with pytest.raises(ValueError, match="Missing required objective metrics"):
        select_best([(score, "missing-token-count")], config)
