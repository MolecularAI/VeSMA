Data vendored from PaRoutes 2.0 for retrosynthesis search benchmarking.

Layout: committed files that tests read live in `processed/`; the large,
downloaded-on-demand raw template library lives in gitignored `raw/`. Scripts at
the dataset root turn raw into processed.

Files (under `processed/` unless noted):
- `n1-targets.smi.gz`, `n5-targets.smi.gz`: gzip-compressed SMILES target sets converted from upstream `targets_n1.txt` and `targets_n5.txt`.
- `n1-stock.smi.gz`, `n5-stock.smi.gz`: gzip-compressed SMILES stock sets converted from upstream `stock_n1.txt` and `stock_n5.txt`.
- `n1-routes.json.gz`, `n5-routes.json.gz`: gzip-compressed reference route trees converted from upstream `ref_routes_n1.json` and `ref_routes_n5.json`.
- `n1-reference-reactions.tsv.gz`, `n5-reference-reactions.tsv.gz`: derived unique reference reaction rows extracted from the route JSON metadata.
- `raw/uspto_raw_template_library.csv.gz`: upstream PaRoutes 2.0 `uspto_template_library.csv`, stored compressed after download and gitignored (not vendored).
- `used14k-reactions.tsv.gz`: committed `(reaction_hash, retro_template)` subset of the raw library, filtered to the 56,657 reaction hashes referenced by n1+n5 routes, which dedup to 13,978 unique templates — codename **used14k** by that unique-template count (formerly named `raw820k-n1n5used56k`). Keeping `reaction_hash` (unlike the deduped `raw68k-templates.tsv.gz`) lets its rule ids align with the route reactions, which route replay/recovery need. Route replay reads this instead of the 116 MB raw library.
- `raw68k-templates.tsv.gz`: committed full raw library deduplicated to its 67,557 unique `retro_template` SMARTS ("raw68k"). The 819,797 raw reaction rows collapse to 67,557 distinct SMARTS, and further to 42,554 distinct `template_hash` values (the curated uspto42k count). `reaction_hash` is dropped from this compact rule artifact; route recovery reconstructs complete alignment by mapping the 56,657 reference hashes retained in `used14k-reactions.tsv.gz` back to raw68k rule IDs by SMARTS. Lets route replay/recovery run the whole raw library without the 116 MB download.
- `raw68k-templates.provenance.json`: manifest recording the raw→unique counts and dedup key for `raw68k-templates.tsv.gz`.
- `uspto_unique_templates.csv.gz`: upstream USPTO50-ish unique retrosynthesis templates used as the search rule set.
- `uspto_unique_templates.compatibility.json`: manifest from checking `uspto_unique_templates.csv.gz` against Vesma's current reaction SMARTS parser.
- `prepare_paroutes.py`: downloads and rewrites the vendored files from Zenodo (into `raw/` and `processed/`).
- `build_used14k_reactions.py`: builds `processed/used14k-reactions.tsv.gz` from the raw library and the n1/n5 routes.
- `build_raw68k_templates.py`: builds `processed/raw68k-templates.tsv.gz` (and its provenance manifest) from the raw library by deduplicating on `retro_template`.

The RDKit/Vesma-compatible filtered subset of these templates lives in the remix
dataset `../remix/uspto42k-compat/processed/uspto42k-compat.csv.gz`.

Attribution and licensing:
- Source: PaRoutes 2.0 and USPTO-based models, Zenodo record `https://doi.org/10.5281/zenodo.7341155`.
- PaRoutes repository: `https://github.com/MolecularAI/PaRoutes`.
- The Zenodo record is published under Creative Commons Attribution 4.0 International.

Derived manifest policy:
- Target and stock text files are stored as `.smi.gz`; line content is not canonicalized during vendoring.
- Reference route JSON is compressed but not otherwise rewritten.
- Reference reaction TSVs are derived from reaction nodes in the reference route JSON files and deduplicated by reaction hash, reaction SMILES, routed reaction SMILES, and patent route ID.
- The full `selected_reactions_all.csv` is intentionally not vendored. It is multi-gigabyte raw data and is not needed for the n1/n5 benchmark search path.
- The benchmark search uses the compatibility-filtered subset in `../remix/uspto42k-compat/processed/uspto42k-compat.csv.gz`, a subset of the compact USPTO template set associated with the PaRoutes 2.0 model.

Compatibility summary for `uspto_unique_templates.csv.gz`:
- Template rows: 42,554
- Rows without recursive SMARTS `$(`: 42,554
- Rows parseable by `vesma.reaction.parse_reaction_smarts`: 42,413
- Rows rejected by Vesma's parser or pattern tightening: 141

(The compatibility summary for the filtered `uspto42k-compat.csv.gz` lives
with that dataset in `../remix/uspto42k-compat/README.md`.)

Current vendored counts:
- n1 targets: 10,000 rows
- n1 stock: 13,432 rows
- n1 derived reference reactions: 31,785 rows
- n5 targets: 10,000 rows
- n5 stock: 13,328 rows
- n5 derived reference reactions: 37,290 rows

Wire formats:
- `*.smi.gz`: one SMILES string per line.
- `*-routes.json.gz`: upstream route tree JSON list.
- `*-reference-reactions.tsv.gz`: tab-delimited `reaction_hash`, `smiles`, `rsmi`, `id`, `ring_breaker`.
- `raw/uspto_raw_template_library.csv.gz`: gzip-compressed upstream raw template library table, tab-delimited (6 columns: `reaction_smiles`, `reaction_hash`, `classification`, `ring_breaker`, `retro_template`, `template_hash`) despite the `.csv` upstream name.
- `used14k-reactions.tsv.gz`: tab-delimited `reaction_hash`, `retro_template`.
- `raw68k-templates.tsv.gz`: tab-delimited `retro_template`, `template_hash`, `library_occurence` (rows sorted by `retro_template`; `library_occurence` is how many raw reaction rows share the SMARTS).
- `uspto_unique_templates.csv.gz`: tab-delimited `template_code`, `retro_template`, `template_hash`, `classification`, `library_occurence`.

Profiling:
- Use a writable cache when profiling so joblib cache hits are measured instead of
  sandbox/cache permission failures:
  `XDG_CACHE_HOME=.profile-cache UV_CACHE_DIR=.uv-cache`.
- Current paper benchmarks are pytest-driven and wired through the repository
  `Makefile` targets such as `bench-route-replay-*`, `bench-route-recovery-*`,
  `bench-match-*`, and `bench-match-plus-split-*`.

## Small PaRoutes Benchmark Corpora

The `subset-retro-benchmark-small-*` directories contain small, test-focused
subsets of the PaRoutes benchmark data. They are generated using the
`tests/data/external/paroutes/extract_small_retro_corpus.py` script.

### Purpose
These corpora are used for fast, automated testing of retrosynthetic route
extraction, replay, and template matching without loading or searching the full
template library. They keep route-level checks tractable while still exercising
multi-step retrosynthesis.

### Generation
Before generating a subset, download the upstream raw template library:

```bash
uv run python tests/data/external/paroutes/prepare_paroutes.py --only uspto-raw-template-library
```

The `raw/uspto_raw_template_library.csv.gz` file is intentionally ignored by
git. Each small subset vendors only the template rows used by its selected
routes.

The `subset-retro-benchmark-small-n1-1` extraction criteria:
1. Routes come from the `n1` reference dataset.
2. Target molecules had routes of exactly `depth = 2`.
3. Every route `reaction_hash` exists in the upstream raw template library.
4. Every route template parses with Vesma's current reaction SMARTS parser.

The `subset-retro-benchmark-small-n5-depth3-4-1` extraction criteria:
1. Routes come from the `n5` reference dataset.
2. Target molecules had routes with `depth = 3` or `depth = 4`.
3. Every route `reaction_hash` exists in the upstream raw template library.
4. Every route template parses with Vesma's current reaction SMARTS parser.
5. The reference route replays to the same molecule identities with both RDKit
   and the Vesma CPU matcher.

Regenerate the n5 subset with:

```bash
uv run python tests/data/external/paroutes/extract_small_retro_corpus.py \
  --route-set n5 \
  --subset-dir subset-retro-benchmark-small-n5-depth3-4-1 \
  --min-depth 3 \
  --max-depth 4 \
  --limit 10 \
  --require-route-replay
```

Data included:
- `<route-set>-routes.json.gz`: Reference routes that met the criteria.
- `<route-set>-targets.smi.gz`: SMILES of the target molecules.
- `<route-set>-stock.smi.gz`: SMILES of the required starting materials.
- `uspto_raw_template_library.csv.gz`: The subset of the raw template library containing only rows whose `reaction_hash` appears in these 10 routes.
