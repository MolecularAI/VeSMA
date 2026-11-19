#!/usr/bin/env bash
# Download the AiZynthFinder public ZINC stock.
#
# **This does not yet produce a .smi, and cannot from this source** -- the upstream file is
# 17.4M InChI keys with no structures, and an InChI key is a one-way hash. See README.md for the
# evidence and for the two routes that would make a ZINC stock usable here.
#
# Reading the file at all needs PyTables, which is not a project dependency. Rather than add one
# for a file we cannot currently use, inspect it transiently:
#
#   uv run --with tables --with pandas python -c "..."
#
# Nothing here is committed: raw/ and processed/ are gitignored.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
url="https://ndownloader.figshare.com/files/23086469"
raw="${here}/raw/zinc_stock_17_04_20.hdf5"

mkdir -p "${here}/raw" "${here}/processed"
curl -fsSL "${url}" -o "${raw}"
echo "wrote ${raw} ($(du -h "${raw}" | cut -f1))"
echo "no .smi derived: the source carries InChI keys only -- see README.md"
