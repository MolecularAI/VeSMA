#!/usr/bin/env bash
# Download the DESP benchmark target sets and split each into targets and starting materials.
#
# Upstream is one line per case, a Python tuple `('<target>', '<starting material>')`, because
# DESP is a *double-ended* planner: a case is solved by a route from that starting material to
# that target. Our search is single-ended to a stock, so `*-targets.smi` alone is a different and
# easier task -- see README.md before comparing any number with the paper's.
#
# Nothing here is committed. The target SMILES derive from Pistachio, a commercial NextMove
# database, so raw/ and processed/ are gitignored and re-fetched rather than vendored.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
url="https://ndownloader.figshare.com/files/47507753"   # benchmarks.zip, 11 kB
raw="${here}/raw/benchmarks.zip"

mkdir -p "${here}/raw" "${here}/processed"
curl -fsSL "${url}" -o "${raw}"
unzip -o -q "${raw}" -d "${here}/raw"

python3 - "${here}" <<'PY'
import ast, sys
from pathlib import Path
here = Path(sys.argv[1])
for name in ("pistachio_hard", "pistachio_reachable", "uspto_190"):
    src = here / "raw" / f"{name}_targets.txt"
    if not src.exists():
        continue
    pairs = [ast.literal_eval(line) for line in src.read_text().splitlines() if line.strip()]
    (here / "processed" / f"{name}-targets.smi").write_text(
        "".join(f"{t}\n" for t, _ in pairs)
    )
    (here / "processed" / f"{name}-pairs.tsv").write_text(
        "".join(f"{t}\t{s}\n" for t, s in pairs)
    )
    # The starting materials as a stock file. A goal-constrained case is recovered when a
    # solution subgraph reaches its own starting material, and a molecule can only end a route
    # if the search can buy it -- so the specified materials have to be loadable as stock.
    materials = dict.fromkeys(s for _, s in pairs)
    (here / "processed" / f"{name}-starting-materials.smi").write_text(
        "".join(f"{s}\tDESP {name} starting material\n" for s in materials)
    )
    print(f"{name}: {len(pairs)} cases")
PY
echo "wrote ${here}/processed"
