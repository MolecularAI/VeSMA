from __future__ import annotations

import shutil
import sys

import pytest

from vesma.metal_matcher import GpuUnavailableError, probe_metal

pytestmark = [pytest.mark.gpu, pytest.mark.metal]


def _skip_if_no_metal_toolchain() -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    if shutil.which("clang++") is None:
        pytest.skip("clang++ is required for the Metal backend.")


def test_gpu_backend_probe_is_available() -> None:
    _skip_if_no_metal_toolchain()

    try:
        info = probe_metal()
    except GpuUnavailableError as exc:
        pytest.skip(str(exc))

    assert info.backend_type == "Metal"
    assert info.name
