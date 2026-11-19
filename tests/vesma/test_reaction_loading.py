from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vesma import reaction
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.cuda_matcher import CudaReactionMatcher

RULE_SMARTS = (
    "[C:1][C:2]>>[C:1].[C:2]",
    "[C:1][O:2]>>[C:1].[O:2]",
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("VESMA_CACHE", raising=False)
    monkeypatch.delenv("VESMA_N_JOBS", raising=False)


def test_reaction_patterns_from_smarts_preserves_sequence_order() -> None:
    patterns = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)

    assert len(patterns) == len(RULE_SMARTS)
    assert patterns[0].rhs.atom_masks.exact_z().tolist() == [6, 6]
    assert patterns[1].rhs.atom_masks.exact_z().tolist() == [6, 8]


def test_reaction_patterns_from_smarts_cache_hit_returns_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)

    def cache_miss(smarts: str):
        raise AssertionError(f"Expected parsed rules to come from cache: {smarts}")

    monkeypatch.setattr(reaction, "parse_reaction_smarts", cache_miss)
    observed = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)

    assert len(observed) == len(expected)
    assert observed[0].rhs.atom_masks.exact_z().tolist() == [6, 6]
    assert observed[1].rhs.atom_masks.exact_z().tolist() == [6, 8]


def test_reaction_pickle_cache_bypasses_joblib_on_warm_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)
    assert reaction.reaction_pickle_cache_path(RULE_SMARTS).exists()

    def no_joblib(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Warm rule loading should use the pickle cache")

    monkeypatch.setattr(reaction, "cached_function", no_joblib)
    observed = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)
    assert [rule.rhs.atom_masks.exact_z().tolist() for rule in observed] == [
        [6, 6],
        [6, 8],
    ]


def test_reaction_patterns_from_smarts_changed_sequence_causes_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)

    call_count = 0
    original = reaction.parse_reaction_smarts

    def tracked_parse(smarts: str):
        nonlocal call_count
        call_count += 1
        return original(smarts)

    monkeypatch.setattr(reaction, "parse_reaction_smarts", tracked_parse)
    reaction.reaction_patterns_from_smarts(tuple(reversed(RULE_SMARTS)), n_jobs=1)

    assert call_count == len(RULE_SMARTS)


def test_reaction_patterns_from_smarts_cache_false_forces_reparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)

    call_count = 0
    original = reaction.parse_reaction_smarts

    def tracked_parse(smarts: str):
        nonlocal call_count
        call_count += 1
        return original(smarts)

    monkeypatch.setattr(reaction, "parse_reaction_smarts", tracked_parse)
    reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1, cache=False)

    assert call_count == len(RULE_SMARTS)


def test_cpu_reaction_matcher_builds_fresh_matchers_from_cached_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)
    first = CpuReactionMatcher(rules)

    def cache_miss(smarts: str):
        raise AssertionError(f"Expected parsed rules to come from cache: {smarts}")

    monkeypatch.setattr(reaction, "parse_reaction_smarts", cache_miss)
    cached_rules = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)
    second = CpuReactionMatcher(cached_rules)

    assert second is not first
    assert isinstance(second, CpuReactionMatcher)


def test_cuda_reaction_matcher_builds_fresh_matchers_from_cached_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_rules: list[Any] = []

    def record_init(self, rules, **kwargs) -> None:
        captured_rules.append(rules)

    monkeypatch.setattr(CudaReactionMatcher, "__init__", record_init)
    rules = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)
    first = CudaReactionMatcher(rules)

    def cache_miss(smarts: str):
        raise AssertionError(f"Expected parsed rules to come from cache: {smarts}")

    monkeypatch.setattr(reaction, "parse_reaction_smarts", cache_miss)
    cached_rules = reaction.reaction_patterns_from_smarts(RULE_SMARTS, n_jobs=1)
    second = CudaReactionMatcher(cached_rules)

    assert second is not first
    assert [len(rules) for rules in captured_rules] == [
        len(RULE_SMARTS),
        len(RULE_SMARTS),
    ]
