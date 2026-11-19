from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests.datadirs import external_data_root


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--bench-mols",
        action="store",
        type=int,
        default=-1,
        help="Number of molecules to use in benchmarks (-1 for all)",
    )
    parser.addoption(
        "--bench-rules",
        action="store",
        type=int,
        default=-1,
        help="Number of rules to use in benchmarks (-1 for all)",
    )


@pytest.fixture
def bench_mols(request: Any) -> int:
    return request.config.getoption("--bench-mols")


@pytest.fixture
def bench_rules(request: Any) -> int:
    return request.config.getoption("--bench-rules")


@pytest.fixture(scope="session")
def readonly_datadir() -> Path:
    """Path to the committed, read-only test-data tree (``tests/data``).

    No copy is made: tests read the files in place. The name signals intent — do not
    write into the returned tree; if a test needs to mutate data, copy what it needs
    into ``tmp_path`` first. The same path is available to non-pytest runners via
    :func:`tests.datadirs.external_data_root`.
    """
    return external_data_root()


@pytest.fixture
def smarts_dataset_dir(readonly_datadir: Path) -> Path:
    return readonly_datadir / "external" / "zbh-smarts"


def pytest_runtest_setup(item: Any) -> None:
    """Skip Metal-marked tests off macOS instead of letting them fail.

    Every Metal test module carries `pytestmark = [pytest.mark.gpu, pytest.mark.metal]`, but the
    actual skip was a helper each module had to remember to call per test. On a Linux CUDA box
    twenty of them did not: five shader-constant tests, fourteen retro tests and one frontier
    keying test failed rather than skipped, which buries a real regression in known noise exactly
    where a GPU suite is the gate -- and makes `make check` useless there.

    The marker is the fact; this makes it do the work. Only the platform is checked, because a
    per-test probe of the device would cost a subprocess for every test on the machine that can
    actually run them. On macOS this does nothing and the modules' own
    `skip_if_no_metal_toolchain` still decides, with a better message for a Mac missing clang++
    or a GPU.
    """
    if item.get_closest_marker("metal") and sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
