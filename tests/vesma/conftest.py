"""pytest-benchmark provenance hooks for the Vesma scaling benchmarks.

Scoped to ``tests/vesma`` so the Vertsy benchmarks are unaffected. These hooks
enrich the JSON that pytest-benchmark writes:

- ``machine_info`` gains full cpu/cuda/metal detail and a combined ``host_type`` slug
  used by the paper's hardware labels.
- ``commit_info.dirty`` is recomputed against ``src/`` and ``tests/`` only, so
  committing result files under ``docs/`` does not flag later runs as dirty.
"""

from __future__ import annotations

import os
from typing import Any

from tests.vertsy import paper_provenance as prov


def pytest_benchmark_update_machine_info(
    config: Any, machine_info: dict[str, Any]
) -> None:
    machine_info["cpu_detail"] = prov.cpu_provenance()
    machine_info["cuda"] = prov.cuda_provenance()
    machine_info["metal"] = prov.metal_provenance(
        probe_backend=os.environ.get("BENCH_PROBE_METAL") == "1"
    )
    machine_info["software"] = prov.software_provenance()
    machine_info["host_type"] = (
        os.environ.get("BENCH_HOST_LABEL", "") or prov.host_slug()
    )


def pytest_benchmark_update_commit_info(
    config: Any, commit_info: dict[str, Any]
) -> None:
    commit_info["dirty"] = not prov.repo_is_clean(paths=("src", "tests"))
    commit_info["clean_scope"] = "src,tests"
    commit_info["source_trees"] = prov.git_provenance()["source_trees"]
