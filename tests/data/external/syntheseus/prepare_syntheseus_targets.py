"""Download the target sets the syntheseus multi-step benchmark searches for.

The syntheseus paper reports multi-step results on the 190 Retro* Hard targets under a
10-minute-per-target limit, so those targets are what an effort-matched comparison needs.
Upstream ships them inside a pickle of reference *routes*; the plain SMILES list vendored
here comes from the retro-fallback repository, which committed the extracted file. The
pickle is deliberately not read: unpickling downloaded bytes executes code, and the SMILES
are all the search needs.

The eMolecules inventory those targets are searched against is 273 MB compressed, so it
downloads on demand into gitignored ``raw/`` rather than being committed.

Run it with ``make dataset-syntheseus-targets``.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DATASET_DIR = Path(__file__).resolve().parent

RETRO_FALLBACK_BASE = (
    "https://raw.githubusercontent.com/AustinT/retro-fallback-iclr24/main/"
    "retro_fallback_iclr24/iclr24_experiments"
)
FIGSHARE_BASE = "https://ndownloader.figshare.com/files"

# Where the 190 targets come from before retro-fallback extracted them. Recorded in the
# provenance manifest so the chain back to Chen et al. (2020) stays legible.
RETROSTAR190_UPSTREAM = f"{FIGSHARE_BASE}/44956066"  # 190-hard-test-routes.pkl
EMOLECULES_INVENTORY = f"{FIGSHARE_BASE}/44956075"  # ORIGINAL-inventory.csv.gz


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    url: str
    subdir: str
    output: str
    mode: Literal["gzip", "copy"]
    # sha256 of the *decompressed* payload, pinned where upstream is a committed file
    # rather than a generated one. A mismatch means upstream moved, not that we mis-read it.
    sha256: str | None = None


DOWNLOADS = (
    DownloadSpec(
        "retrostar190",
        f"{RETRO_FALLBACK_BASE}/retrostar190/retrostar190.smiles",
        "targets",
        "retrostar190.smi.gz",
        "gzip",
        "92ab1b8618bbd5a9edae2257d2c370ac60df08131a52c7cea08c4d245702e079",
    ),
    DownloadSpec(
        "emolecules-inventory",
        EMOLECULES_INVENTORY,
        "raw",
        "emolecules-inventory.csv.gz",
        "copy",
    ),
)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    selected = set(args.only or [spec.key for spec in DOWNLOADS if spec.mode == "gzip"])
    for spec in DOWNLOADS:
        if spec.key not in selected:
            continue
        output = dataset_dir / spec.subdir / spec.output
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and args.skip_existing:
            print(f"skip {output}")
            continue
        download(spec, output)
        if spec.key == "retrostar190":
            write_provenance(spec, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", nargs="?", default=DATASET_DIR, type=Path)
    parser.add_argument(
        "--only",
        choices=tuple(spec.key for spec in DOWNLOADS),
        action="append",
        help="default: every committed target set, but not the 273 MB inventory.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def download(spec: DownloadSpec, output: Path) -> None:
    print(f"download {spec.key} -> {output}")
    tmp = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(spec.url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request) as response:
        if spec.mode == "copy":
            # Streamed: the inventory is 273 MB and never fits a check against a digest
            # we would have to hold in memory anyway.
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        else:
            payload = response.read()
            digest = hashlib.sha256(payload).hexdigest()
            if spec.sha256 is not None and digest != spec.sha256:
                tmp.unlink(missing_ok=True)
                raise ValueError(
                    f"{spec.key}: upstream sha256 is {digest}, expected {spec.sha256}. "
                    "Upstream changed; re-pin only after checking what moved."
                )
            # mtime=0 so re-downloading identical content produces a byte-identical file
            # and does not churn the committed artifact.
            with tmp.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                    gz.write(payload)
    tmp.replace(output)


def write_provenance(spec: DownloadSpec, output: Path) -> None:
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        smiles = [line.strip() for line in handle if line.strip()]
    manifest = {
        "file": spec.output,
        "source": spec.url,
        "upstream": RETROSTAR190_UPSTREAM,
        "upstream_file": "190-hard-test-routes.pkl",
        "targets": len(smiles),
        "unique_targets": len(set(smiles)),
        "sha256_plain": spec.sha256,
        "note": (
            "Retro* Hard: the 190 targets of Chen et al. (2020), the set the syntheseus "
            "paper reports multi-step search on. SMILES are RDKit-canonical as extracted "
            "upstream and are not re-canonicalized here."
        ),
        "computed_on": datetime.date.today().isoformat(),
    }
    path = output.with_name(output.name.split(".")[0] + ".provenance.json")
    print(f"write {path}")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
