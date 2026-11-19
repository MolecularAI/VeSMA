USPTO42k compatible templates (remix).

The PaRoutes USPTO unique templates (`../../paroutes/processed/uspto_unique_templates.csv.gz`)
filtered to the rows whose full reaction SMARTS parse with both RDKit and Vesma's
reaction parser and whose Vesma-parsed LHS fits the GPU molecule representation.
This is the rule set used by the Vesma/RDKit/GPU match and route benchmarks.

Files:
- `processed/uspto42k-compat.csv.gz`: filtered one-column template table.
- `filter_uspto42k_compat.py`: regenerates the processed artifact from the paroutes source. Run via
  `make uspto42k-compat-templates`. Uses the `.venv` interpreter (it has `vesma`).

Compatibility summary:
- Source file: `../../paroutes/processed/uspto_unique_templates.csv.gz`
- Template rows scanned: 42,554
- Rows kept: 42,554
- Rows rejected: 0
- Kept LHS atom count max: 36
- Filter definition: keep rows only when `vesma.reaction.parse_reaction_smarts`
  succeeds, the parsed LHS atom count is at most 64, and RDKit
  `ReactionFromSmarts` succeeds.

Wire format:
- `processed/uspto42k-compat.csv.gz`: gzip-compressed CSV with one column,
  `retro_template`.
