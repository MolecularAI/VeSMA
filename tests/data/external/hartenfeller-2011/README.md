Data downloaded on 2026-05-18 and vendored under this directory for
retrosynthetic template tests and experiments.

Layout:
- `raw/ChemoDOTS_list_of_chemical_reactions.xlsx`: unmodified raw workbook from
  the ChemoDOTS Zenodo v2 record, `https://doi.org/10.5281/zenodo.12093823`,
  file URL
  `https://zenodo.org/records/12093823/files/ChemoDOTS_list_of_chemical_reactions.xlsx?download=1`
**The derived template table has moved to `tests/data/templates/templates-hartenfeller.csv`,
beside the other hand-written tables, and this directory now holds only the upstream workbook.**
It was shelved here because of where its *reaction list* came from, but the table is not vendored
data: every pattern in it was written here -- forward SMIRKS rewritten as retro SMARTS, recursive
SMARTS rewritten flat -- and it was derived manually (with the help of several AI agents/LLMs)
rather than by a script. Filing our own artifact under `external/` implied an upstream source that
could be re-fetched and checked, which was not true of it.

What is genuinely upstream stays: `raw/ChemoDOTS_list_of_chemical_reactions.xlsx`, and the
attribution below, which the moved table's own header repeats. `template_code` still names the
source of each row -- `Hartenfeller 1a`, `iSCB 66b` -- so the two collections remain separable, and
`TEMPLATE_FILES["hartenfeller"]` still measures them as a rule set of their own. The
earlier AiZynthFinder-style `.tsv` derivations and their `convert_to_aizynthfinder.py`
/ `verify_translations.py` helpers have been dropped.

### Hand correction of 2026-09-23: 35 rows whose ring was not written aromatic

Because the table is hand-derived, it is worth recording what a review of every row found
and changed. The defect was systematic rather than scattered: the forward-to-retro
conversion carried the *product* ring over in the Kekule/aliphatic form the forward rule
writes it in (`[C:1]1=[N:2]-[N]=[N]-[N]1`), and a forward rule gets away with that because
the toolkit aromatises the product it builds. A retro template has the opposite job -- its
left-hand side has to *match* a molecule that is already aromatic -- and an aliphatic `C`
or `N`, or an explicit `-`/`=` ring bond, never matches an aromatic atom or bond. Every
such row was therefore dead on arrival.

Measured against the 21,097 distinct products and targets of PaRoutes n1+n5, **35 of 146
rows matched nothing at all**; after correction, 2 do. The two survivors are
`Hartenfeller 1c`/`1d` (Pictet-Spengler onto a ketone, giving a 1,1-disubstituted
tetrahydroisoquinoline): they match a hand-drawn example correctly and are simply rare in
that corpus, so they were left alone.

What changed, all verified through Vesma's own matcher and not only RDKit:

- **Ring written aromatic** -- tetrazole `10a-c`/`11a-c`, Huisgen `12a-c`/`13a-c`/`14a-c`,
  1,2,4-triazole `15`/`16a`/`16b`, pyrazole `19`, Paal-Knorr pyrrole `21`, benzofuran
  `25a-c`, benzothiophene `26a-c`, indole `27a-c`, imidazole `52`. Oxadiazole `28` already
  had aromatic *atoms* but explicit single ring *bonds*, which fails the same way.
- **Fused lactams** -- Niementowski quinazolinone `8` and phthalazinone `20`. Both
  perceive as aromatic including the carbonyl carbon, so both are written with
  `-,:`/`=,:` bonds and `#6`/`#7` atoms, matching either perception.
- **Phthalazinone `20` also lost its carbonyl oxygen.** The left-hand side described a
  plain phthalazine and the right-hand side then invented a carboxylic acid. The oxygen is
  now matched and carried through (map 8).
- **Fischer indole `23` was the worst row and did not show up as dead** -- it matched 1
  molecule in 21,097, and no indole among them. It described an aliphatic 5-ring, and its
  right-hand side wrote `[N:1]-[C:5]=[C:4]`, an enamine, where the reaction needs an
  arylhydrazine. It now yields phenylhydrazine + ketone.
- **Triaryl-imidazole `22`** is the mass-balance row the linter flagged. Two defects: map
  numbers 4 and 5 were given to the two ring nitrogens, which come from ammonium acetate
  and have no precursor, so they were reported as deleted atoms; and `[cH:3]` pinned a
  hydrogen at C2, so the template matched a 4,5-disubstituted imidazole and never the
  2,4,5-trisubstituted one its name promises. The nitrogens are now left unmapped -- which
  is how every other bulk-reagent atom in this table is written, see `10a-c`, `16a/b`,
  `28` -- and C2 is `[c:3]`, which admits both the aryl aldehyde and the formaldehyde case.

**Benzo-fusion is written as an explicit ring, not `;R2`.** Benzofuran, benzothiophene,
indole and Fischer indole all need the halide to sit on the fused six-ring; without that
the five-ring matches in two orientations and one of them tears an aromatic atom out of
every ring. `[c;R2:1]` expresses it compactly but needs SSSR ring perception, which the
Metal and CUDA pipelines switch off, so those rows would mean one thing on the host and
another on a GPU. The benzo ring is instead spelled out atom by atom (maps 6-9), which is
plain connectivity. Those four carbons are re-authored on the right-hand side as well: an
unmapped left-hand atom is deleted by the retro application, which would orphan the ring.

The ChemoDOTS forward rules in `raw/` were used as a cross-check on intent -- which atom
comes from which reagent, and which nitrogens are bulk reagent -- **not as ground truth**.
They carry their own recursive `$(...)` constraints that this table deliberately drops, and
they may be flawed in their own right. Every correction above was accepted on the evidence
that the rewritten row matches real products and returns chemically sensible precursors
through Vesma, not on agreement with the workbook.

### Recovering specificity lost with the recursive SMARTS

Dropping recursive SMARTS left several rows describing far more than the reaction they name.
Where the discarded constraint sits on atoms that **survive into the product**, it can be
spelled out flatly and the row made specific again. Where the constraint was a ring
membership (`r5`, `R0`), spelling the ring out is also the only option here, since the Metal
and CUDA pipelines switch ring perception off. Match rates are against the same 21,097
PaRoutes products.

A rule is **split** wherever one flat template cannot carry a choice of element: Vesma builds
a concrete molecule on the precursor side, so `[#6,#7:6]` is rejected there and each element
needs its own row.

| row | was | now | what was recovered |
| --- | --- | --- | --- |
| `34` Mitsunobu imide | 67.3% | 0.7% | the imide nitrogen, `N(C=O)C=O` |
| `36` Mitsunobu sulfonamide | 67.3% | 2.3% | the sulfonamide nitrogen, `N(C)S(=O)=O` |
| `35` Mitsunobu phenole | 64.6% | 20.4% | the phenol oxygen keeps its aryl |
| `51a-c` Buchwald-Hartwig, N-alkyl | 48.4% | 23.8% | N carries an sp3 carbon |
| `51d-f` Buchwald-Hartwig, diarylamine | 48.4% | 6.3% | N carries a second aryl |
| `53a-c` decarboxylative, ortho-ketone | 28.6% | 4 mols | the ortho acyl directing group |
| `53d-f` decarboxylative, ortho-amide | 28.6% | 31 mols | the ortho acyl directing group |
| `31a-c` Suzuki | 28.6% | 26.8% | boronate carbon flanked by ring carbons |
| `31d-f` Suzuki 2-azinyl boronate | -- | 12.3% | boronate carbon alpha to a ring N |
| `31g-i` Suzuki 2-thienyl boronate | -- | 1.6% | boronate carbon alpha to a ring S |
| `31j-l` Suzuki 2-furyl boronate | -- | 1.4% | boronate carbon alpha to a ring O |
| `18a` Spiro-chromanone, piperidin-4-one | 23 mols | 11 mols | the spiro partner is a six-ring ketone |
| `18b` Spiro-chromanone, cyclohexanone | 23 mols | 0 mols | the spiro partner is a six-ring ketone |
| `49a` N-arylation, pyrrole-type | 6.7% | 0.8% | the azole five-ring |
| `49b` N-arylation, pyrazole-type | 6.7% | 2.2% | the azole five-ring |
| `49c` N-arylation, imidazole-type | 6.7% | 1.4% | the azole five-ring |

**`34` and `36` had become textually identical** -- both were `[C:1]-[N:2]>>[C:1]-[OH].[N:2]`
-- so the loader dropped one as a duplicate. That was not a transcription slip but the
recursion loss erasing the only thing distinguishing an imide from a sulfonamide, and both
are recoverable in full because the nitrogen environment is untouched by the reaction. They
are now distinct and `36` no longer appears in the duplicate list.

**Suzuki `31` is split by what sits next to the boronate carbon, and the original's
constraint was moved rather than copied.** Hartenfeller required *both* coupling carbons to
have two carbon ring-neighbours, which bans pyridin-2-yl Suzuki outright. The chemistry
behind that constraint is protodeboronation, and it applies to the **boron partner only** --
2-chloropyridine is an unremarkable electrophile. In the retro direction `[c:1]` becomes the
boronate and `[c:2]` the halide, so the flank constraint is on `[c:1]` alone and the halide
side is left free. Each alpha-heteroatom case is then its own row, because it is a real and
separately named piece of chemistry: an alpha-azinyl or alpha-furyl boronate needs a MIDA
boronate or ester rather than the boronic acid the template writes. A biaryl whose two
carbons differ still gets both disconnections, since the matcher tries both assignments.

**Buchwald-Hartwig `51` keeps only positive requirements.** Its original amine constraints
are almost all negative -- `!$(N[C,S]=[S,O,N])`, `!$(N[c,O])`, `!$([ND3])` -- and negation of
an environment has no flat form. Enumerating what the two positive rules leave behind shows
nothing worth adding: of the aryl-N bonds they do not cover, 3,289 are amide/carbamate/urea
nitrogens and 482 sulfonamide nitrogens (both excluded by the original on purpose), 2,110 are
`ArNH2`, which would need ammonia as a precursor and is outside this rule set, 890 are N-O,
and 96 are hydrazines. Two positives are the right number.

Narrowed the same way, after measuring **every** row rather than the ones that looked
broad by eye -- which is how the two worst in the file were missed on a first pass:

| row | was | now | what was recovered |
| --- | --- | --- | --- |
| `33a-l` Negishi, by alkyl substitution | 83.0% | 20.3 / 23.3 / 8.8 / 4.7% | no heteroatom on the alkyl carbon |
| `43a-c` Stille vinyl-stannane | 83.0% | 4.5% | the stannane is vinyl, not any carbon |
| `29a-f` / `63a-f` Williamson, by partner | 31.6% | 12.6 / 17.0% | the halide carbon is a CH2 |
| `65a-c` Williamson-like amine | 51.5% | 4.8% | primary amine in, secondary amine out |
| `70` Ester | 27.6% | 20.5% | the ester oxygen carries a carbon |
| `47`, `68a-b` amide | 44.0% | 40.7% | the amide nitrogen carries a carbon |

`33` Negishi is split four ways by what the alkyl carbon carries, because its forward rule
bans *any* S/N/O/P neighbour on that carbon and the only flat way to say so positively is to
spell every heavy neighbour as `[#6]` -- which means knowing how many there are. The aryl
side is deliberately left free: the same ban applied there would forbid 2-chloropyridine, the
Suzuki mistake again.

`43` Stille allowed a vinyl **or** an aryl stannane, and the aryl half is iSCB `66` exactly,
so this row keeps the vinyl half and `66` keeps aryl-aryl.

**Two rows were disconnecting chemistry they do not name, and both produced null steps.**
`70` Ester matched a bare carboxylic acid, `[C:1](=O)-[O:2]` with `[O:2]` an `OH`, and
returned *the same acid plus water*. `29` Williamson matched an ester and offered acetyl
chloride + ethanol, an acylation. Requiring a carbon on the heteroatom fixes both.

**iSCB `63` was Hartenfeller `29` restated with different map numbers.** The loader's
duplicate check compares template text, so a rule renumbered `[C:1]-[O:2]-[C:3]` instead of
`[C:4]-[O:3]-[C:2]` survives as a separate rule however identical its chemistry. `63` was
rewritten with `29`'s numbering, which made the duplicate visible and dropped it at load.

That workaround is gone and so are the rows: `vertsy.template_lint.identical_patterns` decides
the question on the compiled graph through `vesma.pattern_identity.pattern_certificate`, so a
renumbering cannot hide a duplicate and nobody has to hand-match anything. See the correction of
2026-10-13 below.

**Which meant zero-level grouping had to be applied to both rows of such a pair at once.**
Migrating `Hartenfeller 29a/b/c` and `47` to `>>(B).(C)` while leaving `iSCB 63a/b/c` and
`68a` dotted made the texts differ, the dedup stopped firing, and four rules describing one
reaction compiled under two different claims about their own molecularity -- the worse of
both states, and nothing failed while it was true.
`vertsy.template_lint.half_migrated_duplicates` is a standing test over the shipped tables so
that the next one cannot pass unnoticed, and it now decides the question on the certificate
with grouping ignored rather than by stripping parentheses out of the text.

### Correction of 2026-10-13: the fifteen duplicate rows are deleted

This file used to argue the other way, and the argument does not survive being checked:

> Deleting the duplicate rows would be the other way to make this impossible, and it is the
> wrong one: `template_code` names each row's source, which is what keeps the two collections
> separable and `TEMPLATE_FILES["hartenfeller"]` measurable as a rule set of its own.

**The separability it defends was already gone.** All fifteen pairs were byte-identical, so
`load_merged_library` dropped one half of every one of them first-wins. `iSCB 61`, `63a-f`,
`68a` and `69a-f` never compiled; a measurement of "the iSCB rule set" silently excluded every
row it shared with Hartenfeller, which is not a measurement of that rule set. The rows looked
like rules and were not. In `MC 75`/`iSCB 70` the drop went the other way -- this table is read
first, so the code reaching a report was `iSCB 70` while the row doing the work sat in the
medchem table under another name.

So deleting them is a **no-op for what a search compiles**, which is the whole argument for
doing it, and it removes three things keeping them cost:

- the half-migration failure mode described above, which needed every pair kept in lockstep;
- a forward-table error nothing could catch: `FWD H 41d`, the *vinyl*-halide Heck, named
  `iSCB 69b` and `iSCB 69c` among its partners, which are the *aryl*-halide rules. A
  one-example composition check passes that, because the vinyl partners answer for it;
- fifteen rows of maintenance surface that no run ever reads.

Provenance is kept where it belongs: the surviving row's chemistry is unchanged, this file
records which codes were merged into which, and `template_code` still names the source of every
row that is actually a rule.

**The gap this exposed, not yet filled.** `FWD iSCB 68o` -- ester from an acyl chloride and an
alcohol -- named `iSCB 68a` as its only partner, which is `Hartenfeller 47`, an *amide*
disconnection, and a code that never compiled. Wrong chemistry pointing at a dead code, so it
could never confirm a step. The retro table has no "ester to acyl chloride + alcohol" rule,
while `iSCB 67a` is exactly that rule for sulfonate esters -- which makes the carbonyl case look
like an oversight rather than a decision. The forward row is unpaired for now and the retro rule
is unwritten.

Two rows keep a broad left-hand side because there is genuinely nothing to spell:

- **iSCB `66` Stille** -- the forward rule is `[c:1][Sn](C)(C)(C).[Cl,Br,I][c:2]`, which
  constrains neither carbon. A biaryl bond is all it ever described.
- **`53`, `31` and iSCB `66`** still overlap on plain biaryls where their constraints run
  out, but they are three different disconnections of that bond -- carboxylic acid, boronate,
  stannane -- so all are kept and none is a duplicate of another.


**Can Stille really build any aromatic C-C bond?** Close to it, and that is why iSCB `66`
stays broad. Unlike a boronic acid, an aryl stannane alpha to a ring nitrogen is *more*
stable rather than less, so the flank split that Suzuki needs has no analogue here; the real
limits on Stille are tin toxicity and stannane availability, neither of which is a property
of the product.

### `!@`, and why it is applied to some rows and not others

One structural case was wrong: `[c:1]-[c:2]` also matches an aryl-aryl single bond lying
**inside** a ring system, where the disconnection returns a single molecule carrying both the
chloride and the stannane rather than two fragments.

**Bond-level ring/chain queries are supported on every backend and at every retro layer**,
and this is worth stating because it is not true of all the atom-level ones. `;R2` and `;r5`
read one spanning forest's fundamental cycles, whose values depend on its traversal order,
so the Metal and CUDA pipelines leave them unknown. `@`/`!@` needs only per-edge cycle
membership -- a bond is a ring bond iff it is not a bridge -- which does not depend on the
forest. `rings.annotate_bond_topology` computes it when a host `CMol` is packed, and the
Metal and CUDA retro kernels compute it for every precursor they build, so `!@` is a working
constraint on layer one and on layer ten alike. It was not always: until the retro kernels
perceived topology, a device-built precursor declared every bond "ring or chain", and `!@`
matched ring bonds from the second layer down. That is exactly the failure these 51 rows
were written to avoid, so the multi-layer case is now covered by `bench_metal_parity.py` and
`bench_cuda_parity.py`.

It is applied to the **organometallic cross-couplings only** -- `31`, `33`, `43`, `46`, `53`,
iSCB `66`, and `51` -- 51 rows in all. There the single-precursor reading is not chemistry:
it would need an isolable species bearing both an aryl halide and an organometal or boronate
on the same skeleton.

It is deliberately **not** applied elsewhere, and measuring the corpus is what settled that.
An in-ring match for these is a legitimate ring closure whose single precursor is exactly
right, so `!@` would delete real routes rather than noise:

| rule | matches on a ring bond | what that actually is |
| --- | --- | --- |
| `29` Williamson ether | 53.1% | benzodioxole from catechol + `CH2Br2`, THF closure |
| `65` Williamson-like amine | 44.4% | piperidine, pyrrolidine, morpholine closure |
| `30` reductive amination | 32.0% | intramolecular reductive amination, routine ring-forming |
| `47` amide | 22.0% | lactam from an amino acid; macrolactamisation |
| `57` urea | 19.7% | cyclic ureas, hydantoins |
| `70` ester | 3.2% | lactonisation |

### Pinning degree to make an enumeration exhaustive

`51` Buchwald-Hartwig was still wrong after the first pass: `[c:1]-[N:2]-[CX4:3]` requires
*one* good neighbour but leaves the nitrogen's remaining valence free, so an anilide --
`CC(=O)N(C)c1ccccc1`, an amide the forward rule excludes -- still matched. A negation over an
unknown number of neighbours has no flat form, but the count itself can be pinned: `;D2` and
`;D3` fix how many heavy neighbours there are, and then every one of them can be spelled.
The four cases below are exhaustive for a nitrogen bearing an aryl, and `N-methylacetanilide`
no longer matches:

| row | nitrogen | rate |
| --- | --- | --- |
| `51a-c` | `[NH1;D2]` + sp3 carbon -- ArNHR | 7.8% |
| `51d-f` | `[NH0;D3]` + two sp3 carbons -- ArNR2 | -- |
| `51g-i` | `[NH1;D2]` + aryl -- ArNHAr' | 5.6% |
| `51j-l` | `[NH0;D3]` + aryl + sp3 carbon -- ArN(Ar')R | -- |

The same trick is what makes `33` Negishi exhaustive: carbon's valence bounds the split at
four cases (`CH3`, `CH2`, `CH1`, `CH0`), and pinning the hydrogen count pins the heavy-atom
count, so every neighbour can be written `[#6]` and the forward rule's "no S/N/O/P neighbour"
is expressed positively and completely.

Three rows match nothing in the 21,097-product corpus and are kept as correct-but-rare, not
defects: `1c`/`1d` (Pictet-Spengler onto a ketone, giving a 1,1-disubstituted
tetrahydroisoquinoline) and `18b` (spiro-chromanone from a plain cyclohexanone). All three
were checked against hand-drawn examples and return the right precursors.

Attribution and licensing:
- Source data: ChemoDOTS web server data files, Hoffer L.,
  Charifi-Hoareau G., Barelier S., Betzi S., Miller T. W., Morelli X.,
  Roche P., Zenodo, `https://doi.org/10.5281/zenodo.12093823`.
- The Zenodo v2 record is published under Creative Commons Attribution 4.0
  International.
- The ChemoDOTS publication states that its chemical reactions are derived from
  Hartenfeller et al. 2011 and that the complete reaction list and SMARTS
  representations are available from the Zenodo repository.
- Original reaction publication: Hartenfeller M., Eberle M., Meier P.,
  Nieto-Oberhuber C., Altmann K. H., Schneider G., Jacoby E., Renner S.,
  "A collection of robust organic synthesis reactions for in silico molecule
  design", J. Chem. Inf. Model. 2011, 51, 3093-3098,
  `https://doi.org/10.1021/ci200379p`.

What the two collections are, since `template_code` is the only place it is
recorded in the data:

- **Hartenfeller** rows are the 2011 collection above: reactions selected for being
  robust enough to run reliably in a library-synthesis setting, which is why they are
  a common choice for in-silico enumeration rather than a survey of what is possible.
- **iSCB** rows are additions from the laboratory of **integrative Structural and
  Chemical Biology** at the Centre de Recherche en Cancerologie de Marseille (CRCM;
  CNRS UMR 7258, INSERM U 1068, Institut Paoli-Calmettes, Aix-Marseille Universite)
  -- Roche and Morelli's team, the authors of ChemoDOTS. The ChemoDOTS paper describes
  its reaction set as transformations "widely used in the pharmaceutical industry
  during hit-to-lead optimization", so these are the team's own extensions beyond
  Hartenfeller's list rather than a separately published collection. They arrived in
  the same Zenodo record and carry no separate citation.
- ChemoDOTS publication: Hoffer L., Charifi-Hoareau G., Barelier S., Betzi S.,
  Miller T. W., Morelli X., Roche P., "ChemoDOTS: a web server to design
  chemistry-driven focused libraries", Nucleic Acids Research 2024, 52, W461-W467,
  `https://doi.org/10.1093/nar/gkae326`. Server: `https://chemodots.marseille.inserm.fr/`.

Checksums:
- `raw/ChemoDOTS_list_of_chemical_reactions.xlsx` MD5:
  `2c832a02c7136e50fbd060ca347f87f0`
- `raw/ChemoDOTS_list_of_chemical_reactions.xlsx` SHA256:
  `87320b68e59264382ccd7f11d17530bde6e25ec4b0e36329760c3041526f5fa5`

Notes:
- The v2 non-recursive template table is the active benchmark rule set, and all
  rows parse with the current Vesma reaction SMARTS parser.
- **Row counts are deliberately not recorded here.** They changed once already and
  the stale figure outlived the file, which is the failure a committed count invites.
  `wc -l` answers it, and `template_code` separates the two collections:

  ```bash
  gzcat -f processed/hartenfeller_2011_with_iscb_nonrecursive_templates_v2.csv | tail -n +2 | wc -l
  cut -d, -f1 processed/hartenfeller_2011_with_iscb_nonrecursive_templates_v2.csv | sort | uniq -c
  ```

- The table carries a few repeated `retro_template` values. `bench_host_aostar.load_library`
  drops them, and reports `rejected_duplicate` beside `rejected_mass_losing` so the two
  reasons stay distinguishable.

Wire formats:
- `raw/ChemoDOTS_list_of_chemical_reactions.xlsx`: workbook with source columns
  `RULE`, `SMARTS`, `REACTION`.
- `processed/hartenfeller_2011_with_iscb_nonrecursive_templates_v2.csv`:
  comma-delimited `template_code`, `retro_template`, `classification`.
