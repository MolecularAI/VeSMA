Data vendored for CUDA matcher benchmark tests.

This is a remix dataset (see `../README.md`): its committed contract
(`molecules.tsv`, `patterns.tsv`) is derived by combining inputs from two other
datasets in this tree — the ChEMBL 50k subset
(`../../chembl/raw/chembl16_50K_sss.txt.gz`) and the PaRoutes USPTO unique templates
(`../../paroutes/processed/uspto_unique_templates.csv.gz`).

Files:
- `molecules.tsv`: derived deterministic subset of 128 molecules used by the CUDA benchmark.
- `patterns.tsv`: derived deterministic subset of 16 retrosynthetic templates used by the CUDA benchmark.
- `select_subset.py`: rewrites `molecules.tsv` and `patterns.tsv` from the two
  source datasets above using the current pinned benchmark subset.

Attribution and licensing:
- The ChEMBL molecule source is documented in `../../chembl/README.md` (ChEMBL data
  under CC BY-SA 3.0).
- The USPTO template source is documented in `../../paroutes/README.md` (PaRoutes /
  AiZynthFinder, CC BY 4.0).

Derived manifest policy:
- `molecules.tsv` and `patterns.tsv` are the stable benchmark contract consumed by tests and CUDA perf runs.
- `select_subset.py` rewrites the curated subset from the source datasets. The current subset is intentionally pinned so the benchmark keeps the same known false-positive and false-negative examples across refactors.
- The only atom-count filter is the current CUDA implementation limit of 64 atoms.
- `patterns.tsv` is selected by scanning the USPTO unique templates in file order, keeping rows whose full reaction SMARTS and product-side `lhs_smarts` parse with the current Vesma code and whose `lhs_atom_count` is at most 64.
- Positive patterns are the first 8 parseable templates with at least 1 match against the filtered ChEMBL universe.
- Negative patterns are the first 8 parseable templates with 0 matches against the filtered ChEMBL universe.
- `molecules.tsv` contains the pinned 128-molecule subset used for the current retro benchmark, including the disagreement examples discussed in the matcher notes.

Wire formats:
- `molecules.tsv`: tab-delimited `chembl_id`, `smiles`, `atom_count`
- `patterns.tsv`: tab-delimited `template_code`, `template_hash`, `classification`, `library_occurence`, `retro_template`, `lhs_smarts`, `lhs_atom_count`, `hit_class`
