#!/usr/bin/env bash
# Download the ChemoDOTS Enamine building-block stock and derive two files from it.
#
# The upstream table is `smiles<TAB>name`, where `name` is the Enamine catalogue id.
#
#   processed/chemdots_enamine_bbs.smi.bz2            committed. SMILES only, ids dropped.
#   processed/chemdots_enamine_bbs_annotated.smi.bz2  gitignored. SMILES and the id.
#
# Both are written from one download. The committed file is what a fresh checkout searches
# against; the annotated one is what makes a report name the catalogue number a chemist orders
# by, and is re-fetchable rather than vendored because it is an order of magnitude larger.
#
# **The two share one prepared-stock cache.** `open_smiles` keeps the first whitespace-delimited
# column, so both files read as exactly the same tuple of SMILES, and the cache is keyed on that
# tuple rather than on a path -- so adding the annotated file re-keys nothing. There is a test.
#
# No canonicalization, cleanup, or dedup -- see README.md for provenance/licensing
# (Zenodo 10776787, CC BY 4.0).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
url="https://zenodo.org/records/10776787/files/ChemoDOTS_Enamine_BBs.smi?download=1"
raw="${here}/raw/ChemoDOTS_Enamine_BBs.smi"
bare="${here}/processed/chemdots_enamine_bbs.smi.bz2"
annotated="${here}/processed/chemdots_enamine_bbs_annotated.smi.bz2"

mkdir -p "${here}/processed" "${here}/raw"
curl -fsSL "${url}" -o "${raw}"

tail -n +2 "${raw}" | cut -f1        | bzip2 -9 > "${bare}"
tail -n +2 "${raw}" | cut -f1,2      | bzip2 -9 > "${annotated}"

echo "wrote ${bare}"
echo "wrote ${annotated}"
echo "raw download kept at ${raw}; delete it if you only wanted the derived files"
