Target sets for the syntheseus multi-step retrosynthesis benchmark.

Vendored so an AO* run can be pointed at the same molecules the syntheseus paper
searches for, rather than only at PaRoutes n1. Targets alone are not a benchmark: the
inventory they are searched against is a separate 273 MB download (below), and the
single-step model is not reproduced here at all.

Layout: committed files that tests read live in `targets/`; the large,
downloaded-on-demand inventory lives in gitignored `raw/`. The script at the dataset
root fetches both.

Files (under `targets/` unless noted):
- `retrostar190.smi.gz`: the 190 **Retro\* Hard** targets of Chen et al. (2020) — the set
  the syntheseus paper reports all of its multi-step search results on.
- `retrostar190.provenance.json`: manifest recording source URLs, counts, and the pinned
  sha256 of the uncompressed SMILES.
- `raw/emolecules-inventory.csv.gz`: the eMolecules building-block inventory those targets
  are searched against — 23,081,629 molecules, 273 MB compressed, gitignored (not vendored).
- `prepare_syntheseus_targets.py`: downloads the above.

Fetch with:

```bash
make dataset-syntheseus-targets
```

and the inventory, only when a run actually needs it, with:

```bash
uv run python tests/data/external/syntheseus/prepare_syntheseus_targets.py --only emolecules-inventory
```

## What the benchmark spends

The reason this dataset is here — the effort an effort-matched comparison has to match
([Maziarz et al. 2025](https://arxiv.org/abs/2310.19796), §5.2, and `syntheseus/cli/search.py`):

- 600 s (10 minutes) of wall clock per target; `time_limit_s=600` is also the CLI default.
- No cap on single-step model calls (`limit_reaction_model_calls` defaults to 10^6). The
  calls each model actually managed inside the 600 s are reported as a *result*, and range
  from 21 (RootAligned) to 1333, e.g. 113 for MEGAN and 754 for LocalRetro.
- Top-50 predictions per expansion (`num_top_results: int = 50`).
- Search graphs capped at depth 20 for MCTS and 10 for Retro\* — for Retro\* that is routes
  of at most 5 reactions.
- Hyperparameters were tuned on 25 ChEMBL Hard targets at a 300 s limit (see below).

One syntheseus model call expands one molecule into its top-50 disconnections; one Vertsy
AO\* expansion applies the whole template library to one molecule and keeps every success.
Matching on *molecules expanded* is therefore the honest comparison, with the caveat that
our branching is uncapped rather than truncated at 50.

## ChEMBL Hard is not vendored, because it is not published

The syntheseus paper tunes on 25 targets from the ChEMBL Hard set of
[Tripp et al. 2022](https://openreview.net/forum?id=8VLeT8DFeD), but that molecule list does
not appear to be released anywhere: it is absent from `microsoft/syntheseus`, from the
author's `syntheseus-retro-star-benchmark` and `retro-fallback-iclr24` repositories, and from
the figshare record that carries the rest of the retro\* benchmark artifacts. The paper
carries no data or code availability statement. Only the construction is given (§2, verified
against the PDF): 200 molecules chosen randomly from a ChEMBL subset built by applying the
pre-processing script of Brown et al. (2019) — i.e. GuacaMol — keeping only molecules whose
molecular weight, Bertz coefficient, logP **and** TPSA all exceed the mean of the respective
values over the Retro\* Hard set, then removing everything purchasable. The draw is unseeded,
so re-deriving gives a *similar* set, not *the* set.

Re-deriving it would need the raw GuacaMol splits
(`https://ndownloader.figshare.com/files/13612757`, test), descriptor means computed from
`retrostar190.smi.gz`, and the eMolecules inventory to subtract — which is why the
ingredients are recorded here even though the set is not.

Two traps in that paper, worth knowing before matching numbers against it:
- Its figures label this set **"Guacamol Hard"**, not ChemBL Hard. Same 200 molecules.
- Its own budget is 300 s per molecule with a maximum tree depth of 20, and its plots run out
  to 500 reaction-model calls — *not* the 600 s syntheseus later used.

Its other two sets are equally unpublished but cheaper to approximate: 200 molecules drawn at
random from PaRoutes n5 (we vendor all 10,000 of those in `../paroutes/`, so a seeded draw
reproduces the construction), and 200 from GDB17.

## Attribution and licensing

- Targets: Chen et al. (2020), *Retro\*: Learning Retrosynthetic Planning with Neural Guided
  A\* Search*. Distributed as `190-hard-test-routes.pkl` in
  [Syntheseus retro star benchmark data](https://figshare.com/articles/dataset/Syntheseus_retro_star_benchmark_data/25376728)
  (doi `10.6084/m9.figshare.25376728.v1`, MIT).
- The vendored plain-SMILES extraction is the committed
  `retrostar190.smiles` from [retro-fallback-iclr24](https://github.com/AustinT/retro-fallback-iclr24)
  (MIT), which was generated from that pickle. We read the SMILES file rather than the
  pickle on purpose: unpickling downloaded bytes executes code.
- Inventory: `ORIGINAL-inventory.csv.gz` from the same figshare record.

## Derived manifest policy

- SMILES are stored as `.smi.gz`, one per line, exactly as extracted upstream — no
  canonicalization during vendoring, matching the PaRoutes dataset's policy.
- Upstream canonicalized with an older RDKit. Under RDKit 2025.09.3, 189 of the 190 round-trip
  unchanged; the exception is an organotin target whose carbons bonded to `[Sn]` now print as
  `[CH2]`/`[CH]`. All 190 parse. Nothing downstream depends on the stored spelling — the
  keyer canonicalizes on read.
- The prepare script pins the sha256 of the uncompressed SMILES and fails rather than
  overwriting if upstream moves.

## Current vendored counts

- Retro\* Hard targets: 190 rows, all unique, all RDKit-parseable
- Molecular weight: mean 459, range 182–956

## Wire formats

- `targets/*.smi.gz`: one SMILES string per line.
- `raw/emolecules-inventory.csv.gz`: gzip-compressed CSV with header `,mol,idx`; the
  building-block SMILES is column `mol`.
