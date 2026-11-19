#!/usr/bin/env bash
# Download a purchasable ZINC20 stock as SMILES, per tranche, from files.docking.org.
#
# This exists because the ZINC stock AiZynthFinder distributes is 17.4M InChI keys with no
# structures, and an InChI key is a one-way hash -- see ../aizynthfinder/README.md. ZINC20's own
# tranche catalogue is SMILES, so it is the route that leaves the search unchanged.
#
#   processed/zinc20-in-stock.smi.gz            SMILES only. What a search loads.
#   processed/zinc20-in-stock-annotated.smi.gz  SMILES and the ZINC id, for a report.
#
# Both gitignored, both written from the same download. Same pair convention as
# ../chemdots-stock, and for the same reason: `open_smiles` keeps the first whitespace-delimited
# column, so the two read as the same tuple of SMILES and share one prepared-stock cache.
#
# Layout. `https://files.docking.org/2D/<TT>/<TT><r><p>.smi` where `TT` is the molecular-weight
# and logP tranche (AA..KK, 121 of them), `r` is reactivity (A B C E G I) and `p` purchasability
# (A..F). Each file is `smiles zinc_id`, space-delimited, with a one-line header.
#
# Which purchasability codes. `PURCHASABILITY` defaults to `ABCF`, chosen by measurement rather
# than from the code labels: in one mid-size tranche, codes D and E are 1.8 GB and 1.0 GB while
# A, B, C and F together are 8 MB. D and E are make-on-demand space -- the billions ZINC20 can
# enumerate but nobody holds -- and the rest is the catalogue that physically exists. This script
# does not assert which letter means "in-stock" versus "agent"; it asserts only that it is
# excluding the two bulk virtual columns. Override to take all six, or to narrow further.
#
# Reactivity is NOT filtered. An acid chloride is "reactive" to a docking screen and a building
# block to a chemist, and a retrosynthesis stock wants it.
#
# Cost: about 2,900 files, roughly 1 GB uncompressed. Minutes to tens of minutes; six concurrent
# connections, which is polite to a shared academic host. Re-runnable -- `curl -C -` resumes and
# an already-complete file is skipped, so an interrupted fetch costs only what it had not got.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="https://files.docking.org/2D"
raw="${here}/raw"
processed="${here}/processed"
jobs="${JOBS:-6}"
purchasability="${PURCHASABILITY:-ABCF}"

mkdir -p "${raw}" "${processed}"

listing="${raw}/.listing"
mkdir -p "${listing}"

echo "enumerating tranches under ${base}/ ..."
if [[ ! -s "${listing}/root.html" ]]; then
    curl -fsSL --retry 3 --retry-delay 2 "${base}/" -o "${listing}/root.html"
fi
tranches="${TRANCHES:-$(grep -oE 'href="[A-Z]{2}/"' "${listing}/root.html" | sed 's/href="//;s|/"||' | sort -u)}"
echo "  $(echo "${tranches}" | wc -w | tr -d ' ') tranches"

# One listing per tranche rather than a hardcoded reactivity/purchasability alphabet: the
# catalogue is not a full cross product -- tranche AA has 36 files, and a sparse corner of the
# property space has fewer -- so asking the server which files exist avoids a run full of 404s.
echo "enumerating files (purchasability ${purchasability}) ..."
wanted="${raw}/.wanted"
: > "${wanted}"
for tranche in ${tranches}; do
    page="${listing}/${tranche}.html"
    if [[ ! -s "${page}" ]]; then
        curl -fsSL --retry 3 --retry-delay 2 "${base}/${tranche}/" -o "${page}"
    fi
    grep -oE "${tranche}[A-Z]{2}\.smi" "${page}" | sort -u | while read -r name; do
        code="${name:3:1}"
        case "${purchasability}" in *"${code}"*) echo "${tranche}/${name}" >> "${wanted}" ;; esac
    done
done
echo "  $(wc -l < "${wanted}" | tr -d ' ') files"

echo "downloading with ${jobs} concurrent connections ..."
# `--retry` covers a transient 5xx; `-C -` resumes a partial file and exits 0 immediately on a
# complete one, so a re-run is cheap. `|| echo` keeps one missing tranche file from killing the
# sweep -- the count printed afterwards is what actually arrived.
tr '\n' '\0' < "${wanted}" | xargs -0 -P "${jobs}" -I{} bash -c '
    out="'"${raw}"'/2D/{}"
    mkdir -p "$(dirname "${out}")"
    curl -fsS --retry 3 --retry-delay 2 -C - "'"${base}"'/{}" -o "${out}" \
        || echo "missing: {}" >&2
'
got=$(find "${raw}/2D" -name '*.smi' | wc -l | tr -d ' ')
echo "  ${got} files, $(du -sh "${raw}/2D" | cut -f1)"

bare="${processed}/zinc20-in-stock.smi.gz"
annotated="${processed}/zinc20-in-stock-annotated.smi.gz"

# Deduplicated on the SMILES, which the per-tranche split does not do for us: a compound sits in
# exactly one molecular-weight/logP tranche, but the same structure can carry more than one ZINC
# id, and a stock that lists it twice costs a keying pass for no membership it did not already
# have. First id wins. The header line is dropped by refusing a row whose first field is the
# literal `smiles`, which is not a SMILES any catalogue holds.
echo "merging and deduplicating ..."
find "${raw}/2D" -name '*.smi' -print0 \
    | xargs -0 cat \
    | awk '$1 != "smiles" && $1 != "" && !seen[$1]++ { print $1, $2 }' \
    | gzip -9 > "${annotated}"
gzip -cd "${annotated}" | cut -d' ' -f1 | gzip -9 > "${bare}"

echo "wrote ${bare} ($(du -h "${bare}" | cut -f1), $(gzip -cd "${bare}" | wc -l | tr -d ' ') rows)"
echo "wrote ${annotated} ($(du -h "${annotated}" | cut -f1))"
echo "raw tranches kept at ${raw}/2D; delete them if you only wanted the derived files"
