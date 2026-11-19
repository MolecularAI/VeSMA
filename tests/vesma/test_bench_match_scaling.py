from __future__ import annotations

import pytest

from tests.vesma.bench_match_scaling import warmup_rounds


def test_warmup_rounds_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCH_WARMUP_ROUNDS", raising=False)

    assert warmup_rounds() == 1


@pytest.mark.parametrize("value", [0, 3])
def test_warmup_rounds_reads_environment(
    monkeypatch: pytest.MonkeyPatch, value: int
) -> None:
    monkeypatch.setenv("BENCH_WARMUP_ROUNDS", str(value))

    assert warmup_rounds() == value


def test_warmup_rounds_rejects_negative_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCH_WARMUP_ROUNDS", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        warmup_rounds()
