from __future__ import annotations

import hashlib
from pathlib import Path

from tests.vertsy.paper_provenance import (
    parse_core_breakdown,
    parse_ioreg_gpu_core_count,
    parse_system_profiler,
    software_provenance,
    source_artifact,
)


def test_parse_system_profiler_apple_core_variants() -> None:
    text = """
Hardware:

    Hardware Overview:

      Chip: Apple M5 Pro
      Total Number of Cores: 18 (6 Super and 12 Performance)
      Memory: 64 GB

Graphics/Displays:

    Apple M5 Pro:

      Chipset Model: Apple M5 Pro
      Type: GPU
      Total Number of Cores: 20
      Metal: Supported
"""

    info = parse_system_profiler(text)

    assert info["chip"] == "Apple M5 Pro"
    assert info["cpu_core_count"] == 18
    assert info["cpu_core_breakdown"] == "6 Super and 12 Performance"
    assert info["gpu_model"] == "Apple M5 Pro"
    assert info["gpu_core_count"] == 20


def test_parse_core_breakdown_accepts_plain_count() -> None:
    assert parse_core_breakdown("20") == (20, "")


def test_parse_ioreg_gpu_core_count() -> None:
    assert parse_ioreg_gpu_core_count('"gpu-core-count" = 20') == 20


def test_source_artifact_hashes_file(tmp_path: Path) -> None:
    payload = b"corpus-bytes"
    artifact_file = tmp_path / "corpus.smi"
    artifact_file.write_bytes(payload)

    artifact = source_artifact(artifact_file)

    assert artifact["exists"] is True
    assert artifact["bytes"] == len(payload)
    assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()


def test_source_artifact_reports_missing_and_directory(tmp_path: Path) -> None:
    assert source_artifact(tmp_path / "absent.smi") == {
        "path": str(tmp_path / "absent.smi"),
        "exists": False,
    }
    directory = source_artifact(tmp_path)
    assert directory["exists"] is True
    assert directory["type"] == "directory"


def test_software_provenance_records_exact_package_versions() -> None:
    software = software_provenance()

    assert all(
        software[name] for name in ("vesma", "numpy", "pytest", "pytest_benchmark")
    )
    assert software["rdkit"]["available"] is True
    assert software["rdkit"]["version"]
    assert software["rdkit"]["rdkit_version"]
