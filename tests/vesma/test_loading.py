from __future__ import annotations

from pathlib import Path

import pytest

from vesma import loading, parallel


def test_cache_dir_uses_xdg_cache_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert loading.cache_dir() == (
        tmp_path / "xdg" / "vesma" / f"model-{loading.MODEL_SCHEMA}"
    )


def test_cache_dir_falls_back_to_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.chdir(tmp_path)

    assert loading.cache_dir() == (
        tmp_path / ".cache" / "vesma" / f"model-{loading.MODEL_SCHEMA}"
    )


def test_resolve_n_jobs_explicit_arg_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VESMA_N_JOBS", "2")

    assert loading.resolve_n_jobs(1) == 1


def test_resolve_n_jobs_uses_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VESMA_N_JOBS", "3")

    assert loading.resolve_n_jobs(None) == 3


def test_resolve_n_jobs_rejects_zero() -> None:
    with pytest.raises(ValueError, match="must not be 0"):
        loading.resolve_n_jobs(0)


def test_effective_n_jobs_uses_joblib_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel.joblib, "cpu_count", lambda: 4)

    assert parallel.effective_n_jobs(-1) == 4
    assert parallel.effective_n_jobs(-2) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_resolve_cache_uses_env(
    value: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VESMA_CACHE", value)

    assert loading.resolve_cache(None) is expected


def test_resolve_cache_explicit_arg_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VESMA_CACHE", "0")

    assert loading.resolve_cache(True) is True


def test_parallel_map_n_jobs_one_uses_serial_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parallel(*args, **kwargs):
        raise AssertionError("Parallel path should not run for n_jobs=1.")

    monkeypatch.setattr(parallel.joblib, "Parallel", fail_parallel)

    assert loading.parallel_map(lambda value: value + 1, (1, 2), n_jobs=1) == (2, 3)
