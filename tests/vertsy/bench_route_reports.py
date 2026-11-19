"""Render a finished benchmark run directory into a chemist-readable PDF.

Takes the run directory as a parameter. Campaign targets live outside the repository, so
nothing here names a target set or an experiment directory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from vesma.keyer import rdkit_smiles_keyer

from .bench_host_aostar import catalogues_of, open_smiles
from .catalogue import NOTHING, notes_for
from .route_report_data import (
    benchmark_target_archive,
    render_report_qmd,
    route_from_archive,
    route_leaves,
    template_tables,
)

pytestmark = [pytest.mark.perf, pytest.mark.external_data, pytest.mark.very_slow]


def render_pdfs(paths: tuple[Path, ...]) -> None:
    quarto = shutil.which("quarto")
    if quarto is None:
        raise FileNotFoundError("quarto is required to render route PDFs")
    with TemporaryDirectory(prefix="quarto-route-home-") as quarto_home:
        environment = os.environ | {
            "HOME": quarto_home,
            "QUARTO_PYTHON": sys.executable,
        }
        for path in paths:
            subprocess.run(
                [quarto, "render", path.name, "--to", "typst", "--execute"],
                cwd=path.parent,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"{path.stem} PDF ready", flush=True)


def report_catalogues(manifest: Mapping[str, Any]):
    """The run's own stock files, reloaded for provenance, or nothing if they are gone.

    Keyed with the RDKit keyer whatever backend the run used: a prepared catalogue carries both
    keyers' keys, so a lookup hits either way, and this avoids standing up a whole Metal arm to
    answer a question about text.
    """
    paths = [Path(entry["path"]) for entry in manifest["inputs"]["stock"]]
    if not all(path.exists() for path in paths):
        print(
            "stock files are not where the run left them; skipping provenance",
            flush=True,
        )
        return NOTHING
    return catalogues_of(paths, rdkit_smiles_keyer.from_smiles, n_jobs=1)


def write_report(
    archives: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    annotated: bool = False,
    keep_existing: bool = False,
) -> Path:
    """One document for the whole run, and the annotated twin beside it.

    A reviewer reads a run rather than a molecule, and the annotation cycle wants a single file:
    an agent reads this, adds assessments, saves it as the annotated name, and that renders to a
    second PDF. The annotated copy is never overwritten once it exists -- regenerating the
    mechanical report must not discard a chemist's work.
    """
    if keep_existing and destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report_qmd(archives, annotated=annotated))

    return destination


def run_archives(root: Path) -> tuple[Path, tuple[dict[str, Any], ...]]:
    """Turn a run directory into per-target archives, without re-searching or re-replaying.

    The archive is what the renderer reads and what an annotation refers to, so it is written to
    disk rather than kept in memory: a report can then be re-rendered, re-styled or re-filtered
    from a finished run in seconds.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    templates = json.loads((root / "templates.json").read_text())
    records = [
        json.loads(line) for line in (root / "targets.jsonl").read_text().splitlines()
    ]
    # Joined on the target as the sweep was given it, not on line order: `alternatives.jsonl` has
    # a line only for a target that reported routes, so the two files are not parallel. Absent for
    # a run made before the report existed, which still has to render.
    alternatives_file = root / "alternatives.jsonl"
    alternatives = (
        {
            entry["input_smiles"]: entry
            for entry in (
                json.loads(line)
                for line in alternatives_file.read_text().splitlines()
                if line.strip()
            )
        }
        if alternatives_file.exists()
        else {}
    )
    smarts, names = template_tables(templates)
    # Provenance is built *here*, after the search, not carried through it. The search wants a
    # Collection and holds one frozenset; a report wants a Mapping and can afford every file's
    # keys separately, which is the only way a molecule three suppliers carry is reported three
    # times. About 11 s warm against a render that takes minutes.
    catalogues = report_catalogues(manifest)
    # Through `open_smiles`, which handles the gzipped catalogues: a targets file is as likely
    # to be `.smi.gz` as `.smi`, and `read_text` on one fails with a decode error rather than
    # anything that names the problem.
    supplied = list(open_smiles(Path(manifest["inputs"]["targets"]["path"])))
    archive_dir = root / "route-archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archives = []

    for index, record in enumerate(records, start=1):
        metadata = {
            "target_index": index,
            # The line as supplied, which preprocessing may have changed into what was searched.
            "input_target": (
                supplied[index - 1] if index <= len(supplied) else record["target"]
            ),
            "stock": ", ".join(entry["path"] for entry in manifest["inputs"]["stock"]),
            "templates": manifest["template_library"],
            "source": str(root / "targets.jsonl"),
            "preprocessing": manifest["target_preprocessing"],
            "template_smarts": smarts,
            "template_names": {str(rule): name for rule, name in names.items()},
        }
        # Through the archive's own reader rather than a second walker over the same JSON:
        # the node field is `smiles` and a leaf is one with no template, and knowing that twice
        # is how the two drift apart.
        leaves = {
            smiles: rdkit_smiles_keyer.from_smiles(smiles)
            for node in record["route_records"]
            for smiles in route_leaves(route_from_archive(node))
        }
        archive = benchmark_target_archive(
            record,
            metadata | {"stock_sources": notes_for(catalogues, leaves)},
            alternatives.get(metadata["input_target"]),
        )
        (archive_dir / f"target-{index:02d}.json").write_text(
            json.dumps(archive, indent=2) + "\n"
        )
        archives.append(archive)

    return archive_dir, tuple(archives)


def test_saved_host_route_reports() -> None:
    """Render a finished run directory into one PDF, and its annotated twin if present."""
    configured = os.environ.get("HOST_ROUTE_RUN_DIR")

    if not configured:
        pytest.skip("set HOST_ROUTE_RUN_DIR to a finished host benchmark run")

    root = Path(configured).resolve()
    _, archives = run_archives(root)
    selected = os.environ.get("HOST_ROUTE_TARGETS")

    if selected:
        wanted = frozenset(map(int, selected.split(",")))
        archives = tuple(a for a in archives if a["target_index"] in wanted)

    assert archives, "the run directory holds no target records"

    report = write_report(archives, root / "report.qmd")
    write_report(
        archives, root / "report-annotated.qmd", annotated=True, keep_existing=True
    )
    render_pdfs((report,))
