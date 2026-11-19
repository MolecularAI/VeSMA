Classified and curated raw68k templates (remix).

The PaRoutes `raw68k` template library (`../../paroutes/processed/raw68k-templates.tsv.gz`)
joined to reaction-class labels from two independent classifiers, with a per-template verdict
saying whether the two agree. The point is to be able to tell a well-formed, confidently
classified template from an ambiguous one — neither AiZynthFinder's nor PaRoutes' template file
supports that, because both carry a `classification` column filled with `0.0 Unrecognized` for
every row. Populated, parseable, and meaningless.

Everything the build needs is committed. `make raw68k-classified` reads `vendored/` and the
PaRoutes templates, touches no network, runs no classifier, and finishes in seconds. The
expensive paths — a 106 MB library download, a ~3.5 h classifier run — exist as optional targets
that nothing depends on.

Layout: committed artifacts in `processed/`, committed label sources in `vendored/`,
regeneration-only downloads in gitignored `raw/`.

Files:
- `processed/raw68k-classified.tsv.gz`: all 67,557 raw68k SMARTS with labels and a verdict. The primary artifact.
- `processed/raw68k-<verdict>.tsv.gz`: **not committed.** A pure projection of the file above, built by `make raw68k-curated` in about a second. `raw68k-agree.tsv.gz` is the curated rule set.
- `processed/class-mapping.tsv`: the ReactionClassifier→Rxn-INSIGHT class mapping the verdict uses, with per-class purity.
- `processed/raw68k-classified.provenance.json`: counts, sources and mapping depth for the build.
- `vendored/reactionclassifier-labels.json.gz`: ReactionClassifier's label per `template_hash`, with its vote count, agreement and symbolic-confirmation count.
- `vendored/rxn-insight-labels.json.gz`: the same from Rxn-INSIGHT.
- `vendored/zenodo-labels.tsv.gz`: one label per `template_hash` projected from ReactionClassifier's published 665,901-reaction corpus, used as a third opinion.
- `vendored/taxonomy.json.gz`: the 14,060-entry code→name taxonomy, without which the codes are unreadable.
- `manual-overrides.tsv`: three hand-assigned labels, disclosed below.
- `build_raw68k_classified.py`: regenerates the committed artifacts in `processed/` from the files above.
- `filter_curated.py`: projects the classified table down to one verdict. `--verdict unconfirmed` shows what curation removed.
- `classify_reactionclassifier.py`, `classify_rxn_insight.py`: regenerate `vendored/`. Not on the build path.

## The three subsets of raw68k

`uspto42k` and `used14k` are two *different, overlapping* subsets of raw68k, not dedup views of
it, and both are carried as columns rather than as separate files:

| set | SMARTS | note |
|---|---|---|
| raw68k | 67,557 | all of PaRoutes |
| `in_uspto42k` | 42,554 | one chosen spelling per `template_hash` |
| `in_used14k` | 13,978 | the templates the n1/n5 reference routes use |
| both | 9,725 | neither set contains the other |
| neither | 20,750 | spellings uspto42k did not pick and routes did not use |

**Read occurrence, not row counts.** `used14k` is 27.4% of hashes but **73.6% of
`library_occurence`** — route-used templates are the frequent ones, so a set that looks small by
row count can still cover most real chemistry. Every count below is therefore given both ways.

## How the labels cover the library

Labels are keyed on `template_hash`, so all spellings of one template share a label. That is
correct — they are one transformation. Coverage is total: raw68k's hash set and the label files'
hash set are *identical*, so all 67,557 SMARTS are labelled through 42,554 hashes. 25,061 hashes
have a single spelling; the most-spelled has 15.

| verdict | SMARTS | hashes | occurrence |
|---|---|---|---|
| `agree` | 42,737 | 25,284 | **74.6%** |
| `mapping-conflict` | 7,549 | 4,679 | 18.8% |
| `unconfirmed` | 17,053 | 12,478 | 5.9% |
| `zenodo-conflict` | 218 | 113 | 0.8% |

A template is `agree` when ReactionClassifier symbolically confirmed it, neither label is
`0.0 Unrecognized`, its class maps to the Rxn-INSIGHT name actually observed, and the published
corpus does not contradict it. Rejects are kept with a `verdict_reason` rather than dropped, so
the curation can be inspected and argued with.

`unconfirmed` is the largest reject class by count but the smallest by occurrence (5.9%):
confirmation fails mostly on rare templates. `zenodo-conflict` is tiny, which is itself a result
— see below.

**An independent check that curation tracks template quality.** `vertsy.template_lint` finds
templates whose product side maps an atom the precursor side never mentions — RDKit deletes such
an atom and everything attached to it, so one can "solve" a complex target in two steps by
discarding most of it. That defect is found by reading the SMARTS text and has nothing to do
with reaction classes. Yet:

| | templates | mass-losing |
|---|---|---|
| raw68k | 67,557 | 71 |
| curated (`verdict == "agree"`) | 42,737 — 63.3% | **4** |

Curation keeps 63% of the library and removes **94%** of the known-defective templates, without
being designed to. Two unrelated measures of "this template is wrong" agree, which is the closest
thing here to external validation of the verdict.

**Using the curated set as a rule set.** `make raw68k-curated` writes
`processed/raw68k-agree.tsv.gz`, a gzipped TSV with a `retro_template` column — the format
every rule-set reader here already takes (`bench_host_aostar.read_templates`,
`paroutes_reference.raw68k_rule_index`). It loads directly; the extra columns ride along and
are ignored. It is a build product rather than a committed file because it is a pure function
of `raw68k-classified.tsv.gz`, and tracking a function of a tracked file means re-checksumming
it on every change to the verdict rule.

## Do a template's three reactions agree with each other?

Each template is labelled by majority vote over exactly three of its own USPTO reactions, so
self-consistency is measurable for free. It differs sharply between the two:

| agreement across a template's 3 reactions | ReactionClassifier | Rxn-INSIGHT |
|---|---|---|
| unanimous (3/3) | 32,108 — **75.5%** | 41,320 — **97.1%** |
| split (2/3) | 7,895 — 18.6% | 1,198 — 2.8% |
| all three differ | 2,548 — 6.0% | 33 — 0.1% |

**That is not evidence Rxn-INSIGHT is the better classifier**, and the comparison is not
apples to apples: Rxn-INSIGHT picks from 12 coarse names while ReactionClassifier emits a leaf
code from a 6,962-class hierarchy, so it has vastly more ways to disagree with itself. The
stored `agreement` is at leaf granularity for both, which flatters the coarser vocabulary. The
same pattern shows up wherever this has been measured: fine-grained codes disagree often,
superclasses rarely.

This is why the verdict keys on **symbolic confirmation** rather than on unanimity —
confirmation asks whether a template actually reproduces the recorded product, which is a
claim about chemistry rather than about how finely the vocabulary slices it. `rc_agreement`
and the vendored audits are carried so a stricter rule can be built later, but requiring
unanimity from both would cut the set to 48.3% of hashes for a reason that is mostly an
artifact of taxonomy depth.

## The class mapping

The two classifiers speak different vocabularies: ReactionClassifier emits a hierarchical code
(`2.1.2.4`), Rxn-INSIGHT one of 12 coarse names (`Acylation`). "Agreement" therefore needs a
mapping, derived empirically as the majority Rxn-INSIGHT name per ReactionClassifier class on the
confirmed subset, and committed as `class-mapping.tsv`.

Built at **depth 2**, giving 70 classes at weighted purity **0.844**. Depth 1 is too coarse where
the taxonomy is itself mixed — superclass 5 lumps Deprotections, Reductions and Protections
together and sits at 54% purity — and depth 2 splits `5.1`/`5.2`/`5.3`, taking survivors from
55.9% to 59.7% of hashes.

Low-purity classes are listed rather than dropped, because low purity here is mostly a property
of the taxonomy rather than a mapping error — a catch-all cannot map purely onto one name:

| class | taxonomy name | maps to | purity | templates |
|---|---|---|---|---|
| `7.8` | Other Oxidiation | Acylation | 32% | 276 |
| `8.7` | Other FGI | Miscellaneous | 36% | 682 |
| `7.6` | Benzylic/allylic oxidation | Heteroatom Alkylation and Arylation | 41% | 56 |
| `2.8` | Other Acylation and related processes | Miscellaneous | 48% | 112 |
| `9.4` | Other FGA | Heteroatom Alkylation and Arylation | 49% | 61 |
| `4.2` | O-containing | Heteroatom Alkylation and Arylation | 50% | 195 |

Four of the six are "Other …" buckets. `7.6` and `4.2` are not, and are worth a second look if
this mapping is ever tightened.

**The taxonomy is LLM-derived, not NextMove/RXNO**, despite the NextMove-like shape of its codes.
Its superclass numbering differs: `6` is Reductions, `7` Oxidations, `8` functional group
interconversion, `9` functional group addition, and there is no `10` or `11`. Reading the codes
as RXNO gets the chemistry wrong by roughly one place from 5 onward.

## The published corpus as a third opinion

ReactionClassifier publishes the corpus behind its gate model: 665,901 reactions, of which
`source_library` is `uspto` for 628,870 — exactly the gate's reported `n_train`. It carries
`template_hash` and `retro_template`, so it joins to raw68k directly.

Measured coverage and agreement:

- Joins **19,736 of 42,554 hashes (46.4%)** and 31,206 of 67,557 SMARTS (46.2%) — but **89.3% of
  `library_occurence`**, so the covered templates are the frequent ones.
- Every covered hash carries a single label across all its reactions (100% unanimous), so the
  label is a property of the template there, not of the reaction.
- Against our independently re-derived ReactionClassifier labels it agrees on **98.7%** of
  superclasses, 98.2% at level 2 and 96.8% at leaf; on the symbolically confirmed subset,
  99.2 / 98.7 / 97.5%.

That last figure is the useful one: two independent runs of the same classifier, one by its
authors over their own corpus and one here over PaRoutes reactions, agree on 98.7% of
superclasses. It is used only as a veto — a template whose published label contradicts ours at
depth 2 is marked `zenodo-conflict` — which fires on 113 hashes, 0.8% of occurrence.

**Do not try to join `class_to_templates.json` from the wheel.** Measured: its 44,449 templates
overlap raw68k by **zero**, in both directions. It is the `rr0rp1_ring0` extraction variant, a
different radius from PaRoutes' `retro_template`, and the published corpus is what joins instead.

## Hand-assigned labels

Three templates — `2110`, `24283`, `40914`, `library_occurence` 18, 9, 18 — are the only ones
**both** classifiers return `0.0 Unrecognized` for, and they are the same three rows in both
files. They are one chemistry: the Balz-Schiemann reaction, a primary aromatic amine to an aryl
fluoride via a diazonium salt. What most likely defeats both is the fluorinating agent's SMILES,
written `F-[PH](-F)(-F)(-F)(-F)-F` rather than as a tetrafluoroborate anion.

They carry `9.1.5.12` *Aryl Primary Amine to Aryl Fluoride*, taken from the vendored taxonomy
rather than invented — see `manual-overrides.tsv` for why that code and not the synonymous
`8.7.4.18`. Every row they produce is marked `label_source = "manual"`, so the curated set can be
built with or without them. Three hashes, four SMARTS rows.

This is the only part of the dataset not derived by a script, and it is disclosed here for the
same reason `../../hartenfeller-2011/README.md` discloses its manual table.

## Attribution and licensing

- Templates and reactions: **PaRoutes 2.0**, Zenodo `https://doi.org/10.5281/zenodo.7341155`,
  repository `https://github.com/MolecularAI/PaRoutes`, CC BY 4.0.
- **ReactionClassifier 0.1.1**, MIT, `https://github.com/schwallergroup/ReactionClassifier`.
  `vendored/taxonomy.json.gz` is its bundled data, redistributed under that licence.
- Published corpus: Zenodo record `21097758`, file `reaction_db_public.parquet`, sha256
  `af8482038abf24adc91e33dfd009392c61b9629dca4c2f5aff96d53295390549`. NameRXN-derived columns are
  excluded upstream, so nothing proprietary is redistributed here.
- **Rxn-INSIGHT**, MIT, `https://github.com/mrodobbe/Rxn-INSIGHT`; Dobbelaere, Lengyel, Stevens
  and Van Geem, *J. Cheminform.* 16, 37 (2024).

## Derived manifest policy

- Labels are keyed on `template_hash` and applied to every spelling of that hash.
- The mapping is derived from the confirmed subset only and committed, not recomputed by readers.
- Rejected templates are kept with a reason; only the `raw68k-<verdict>.tsv.gz` projections are filtered, and those are build products.
- gzip mtime is pinned to 0 so the artifacts are byte-reproducible and the checksums below are
  falsifiable. Rebuilding must reproduce them exactly.

## Counts and checksums

- raw68k SMARTS: `67,557` over `42,554` hashes.
- Labelled by both classifiers: `42,554` hashes (100%).
- Symbolically confirmed: `30,073` hashes.
- Mapping: `70` classes at depth 2, weighted purity `0.8444`.
- Agree: `25,284` hashes, `42,737` SMARTS, `74.6%` of `library_occurence`.
- Hand-assigned: `3` hashes, `4` SMARTS rows.

```
0e067c93a25b0c2a7d4cb26359caa212b8c837a53907ac1b6043793d9860b90e  vendored/reactionclassifier-labels.json.gz
ec4d670329794e7d433ea738d977cdd7d34eab676d527642c140488d07dc51f9  vendored/rxn-insight-labels.json.gz
d2558f07a2750e02f418b3503080caf1c98ec617b2a9e8ca4c9f44af7f5fc15c  vendored/taxonomy.json.gz
dbfbcacfd9c4213aebfc0972c35be7a7f2028f16d871259b173b037085f1d12d  vendored/zenodo-labels.tsv.gz
2ececd771b2bcefbb5f82aaf1c91b2c5ae46f45c37232b39e8331d041fbc04b4  manual-overrides.tsv
52135b657dbd24d801ec4dcec4155c2a3637b8122e007fb95044867dd16ac9b6  processed/class-mapping.tsv
45113f4edcbcd2c99061583a8173771886d70c60ac56b87d88d94d98921fe6dd  processed/raw68k-classified.tsv.gz
```

The projection is not listed: it is not committed. `make raw68k-curated` reproduces it
byte-for-byte anyway, since it pins gzip's mtime the same way.

## Wire formats

- `raw68k-classified.tsv.gz` and its `raw68k-<verdict>.tsv.gz` projections: gzip-compressed TSV, columns
  `retro_template`, `template_hash`, `library_occurence`, `rc_code`, `rc_agreement`,
  `rc_verified_votes`, `ri_name`, `mapped_ri_name`, `zen_code`, `verdict`, `verdict_reason`,
  `label_source`, `in_uspto42k`, `in_used14k`. Sorted by `retro_template`.
- `class-mapping.tsv`: TSV, `rc_class`, `ri_name`, `purity`, `templates`.
- `vendored/*-labels.json.gz`: gzip-compressed JSON list, keys `template_hash`,
  `classification`, `votes`, `agreement`, `verified_votes`.
- `vendored/zenodo-labels.tsv.gz`: TSV, `template_hash`, `zen_code`, `zen_reactions`.
- `vendored/taxonomy.json.gz`: gzip-compressed JSON object, code → name.
- `manual-overrides.tsv`: TSV with `#` comments, `template_hash`, `code`, `name`, `basis`,
  `added_on`.

## Regeneration

```bash
make raw68k-classified            # the normal path: committed inputs, seconds
make raw68k-curated               # the agree-only rule set, ~1 s
make raw68k-classified-download   # optional: 106 MB library + 211 MB corpus
make raw68k-classified-classify   # optional: reruns both classifiers, hours
```
