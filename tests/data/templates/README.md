Hand-written rule sets and stock, as opposed to the vendored collections under
`../external/`. Nothing here was downloaded; every row was written for this repository
and is verified by `tests/vertsy/test_hand_written_templates.py`.

## `templates-hartenfeller.csv` -- our patterns, a borrowed reaction list

This table used to sit under `../external/hartenfeller-2011/processed/`, which misdescribed it.
Nothing in it is vendored data: every pattern was written here, forward SMIRKS rewritten as retro
SMARTS and recursive SMARTS rewritten flat, and the table was derived manually rather than by a
script, so it could never be re-fetched or re-checked against an upstream the way a vendored file
can. Only the *reaction list* is borrowed -- which reactions are worth having -- and that is worth
acknowledging precisely because the rest is not.

Provenance of that list: Hartenfeller M., Eberle M., Meier P., Nieto-Oberhuber C., Altmann K.-H.,
Schneider G., Jacoby E., Renner S., *J. Chem. Inf. Model.* 51, 3093-3098 (2011), plus the iSCB
additions from the ChemoDOTS reaction collection, Zenodo v2 record
`https://doi.org/10.5281/zenodo.12093823`. The unmodified upstream workbook stays at
`../external/hartenfeller-2011/raw/ChemoDOTS_list_of_chemical_reactions.xlsx`, with that
directory's README recording the derivation and the hand corrections made to it.

`template_code` still names each row's source -- `Hartenfeller 1a`, `iSCB 69e` -- so the two
collections remain separable, and `TEMPLATE_FILES["hartenfeller"]` still measures them as a rule
set of their own. That mattered for keeping the move honest: merging the rows into
`templates-medchem.csv` would have destroyed the ability to measure one library against another,
which several benchmarks rely on, and would have bought no naming consistency since the codes stay
as they are either way.

Numbered targets below -- "target 15", "targets 5 and 9" -- are positions in a campaign
target set that is **not in this repository**. It lives in a parallel directory and is passed
to the bench as `AOSTAR_HOST_TARGETS_FILE`; the numbers are kept because they are how the
rows were reviewed, and each is described where it first appears. Nothing in the code reads
that directory, and the bench defaults to a vendored PaRoutes subset instead.

All template files use the AiZynthFinder-compatible column layout the Hartenfeller table
uses -- `template_code`, `retro_template`, `classification` -- so they merge with it directly:

```
AOSTAR_HOST_RULE_SET=hartenfeller,medchem,selenium,carbohydrate,protecting-groups
AOSTAR_HOST_STOCK=enamine,chelators,selenium
```

No rule here contains an SSSR primitive (`;R2`, `;r5`), because the Metal and CUDA
pipelines switch ring perception off. Bond-level `@` and `!@` are used freely: those need
only per-edge cycle membership, which `rings.annotate_bond_topology` computes and which
both GPU paths request.

## `templates-selenium.csv`

Written for a cyclic selenonium target,
`O[C@H]1[C@H](O)C[Se@@+](C[C@H]([C@@H]2O[C@H](C3=CC=CC=C3)OC[C@H]2O)O)[C@@H]1CO`, a cyclic
**selenonium salt** -- a selenolane whose selenium is trisubstituted and positively charged,
carrying a sugar-derived arm. It is a salacinol/blintol analogue. No vendored collection
here carries a single selenium rule.

`Se 1` is the disconnection that matters: a selenonium salt comes from a neutral selenide
plus an epoxide, and applied to target 15 it returns the selenolane diol
(1,4-anhydro-4-seleno-arabinitol) plus the benzylidene-protected sugar epoxide, which is
the literature route. `Se 2a-c` is the same bond from an alkyl halide instead.

Two details are easy to get wrong and are worth recording:

* **The precursor selenium must be written `[Se;H0;+0:1]`, with both the charge and the
  hydrogen count stated.** The three spellings behave differently, and only one of them
  tells you when it is wrong:

  | right-hand side | result |
  | --- | --- |
  | `[Se:1]` | **silently wrong**: the product's `+1` carries over, giving `[SeH+]`, a protonated selenonium rather than a selenide |
  | `[Se+0:1]` | raises `UnsupportedReactionError: charge_changed_with_h_omitted` |
  | `[Se;H0;+0:1]` | correct -- a neutral dialkyl selenide |

  The middle case is the only loud one. Stating the charge alone makes Vesma demand the
  hydrogen count; stating neither leaves the charge untouched and hands back a precursor
  that looks plausible and is not.
* **The bond being broken is marked `!@`.** Without it the matcher also breaks a ring bond
  of the selenolane, returning a single mangled molecule alongside the correct answer.

## Macrocyclisation rules

`Mac 1`-`Mac 5`, in `templates-medchem.csv`. They are the ring-bond spelling of disconnections
medchem also carries acyclic: `Mac 5` is `Cpl 1` with `-@` where `Cpl 1` has `-!@`, `Mac 3` and
`Mac 4` are the Suzuki and Stille pair, `Mac 1` and `Mac 2` are alkene and alkyne ring-closing
metathesis.

A two-component retro template closes rings on its own. When the bond it breaks lies in a cycle,
the two right-hand components stay connected through the rest of the ring, so component separation
returns one precursor carrying both halves -- `MC 72` on a macrocyclic triazole returns a single
chain bearing an azide and a terminal alkyne, which is ring-closing click without a rule written
for it.

### Minimum ring size, without SSSR

`Mac 1`, `Mac 2` and `Mac 5` require a minimum ring size, written as an explicit `-@` chain of the
ring atoms the disconnection needs:

| rule | floor | why |
|---|---|---|
| `Mac 1` ring-closing metathesis | 5 | cyclopropene and cyclobutene are strained; five and up is routine |
| `Mac 2` ring-closing alkyne metathesis | 8 | cyclooctyne is the smallest isolable cycloalkyne |
| `Mac 5` macrolactamisation | 5 | a beta-lactam does not form by simple cyclisation |

This needs only per-edge cycle membership, so it works where `;r5` cannot. `Mac 5`'s floor rejects
both an isolated beta-lactam and penicillin G, whose beta-lactam is fused.

`Mac 3` and `Mac 4` carry no floor. Their matched atoms are aromatic, so the `-@` path runs inside
the benzene ring rather than around the macrocycle: a six-atom floor rejects a legitimate
seven-membered cyclophane along with fluorene. The technique measures "this many ring atoms are
reachable", which equals ring size only when the matched atoms belong to one ring.

Larger rings than the floor are not restricted, and forming a small ring rather than a macrocycle
is a valid disconnection -- ring-closing metathesis to a cyclohexene and lactamisation to a
six-membered lactam are both real chemistry.

## Fixed: a bridged ring system flagged an atom aromatic off every aromatic bond

Found through a **cpu**-arm run on n1 target 41, which died with `ValueError: CMol aromatic atoms
must belong to aromatic bonds`, and bisected to one forward rule -- `FWD H 51g`, the Buchwald
N-arylation `[c:1]-[Cl,Br,I].[NH2;D1:2]-[c:3]>>[c:1]-[N:2]-[c:3]` -- on one substrate.

| substrate | ring the N closes | before | after |
| --- | --- | --- | --- |
| `Nc1ccccc1-c1ccccc1Cl` | five, carbazole | aromatic N | unchanged |
| `Nc1cccc(-c2ccccc2Cl)c1` | six | **raised** | aliphatic N |
| `Nc1ccc(-c2ccccc2Cl)cc1` | seven | **raised** | aliphatic N |
| `Nc1ccccc1.Clc1ccccc1` | none, intermolecular | works | unchanged |

**It was never about the forward probe, or about reactions at all.** The same molecule written
directly -- `N1c2cccc(c2)-c2ccccc21` -- could not be read either, and RDKit reads it fine, 12
aromatic atoms over 12 aromatic bonds. Any path that reaches perception reached the defect: SMILES
reading, retro application, `cmol_from_gmol`. A three-molecule retro batch containing one such
molecule died whole, because the bare `ValueError` is not a `ChemistryRejection` and so escaped the
per-application containment in `cpu_matcher.py`.

### What it was

`perceive_aromaticity` assigns the flag positively; nothing is carried over from the input. In the
**joint-acceptance** branch -- entered when a fused component's rings do not each carry an
acceptable count on their own, but the component does as a whole -- the atoms were taken from the
union of every cycle in the component while the edges were taken from the perimeter alone. Those
are different sets and nothing reconciled them.

A nitrogen bridging a biaryl's *meta* positions is where they came apart. The component holds four
cycles: both benzenes, a six-ring through the nitrogen the short way round, and the dependent
eight-ring the long way. The benzenes count 6 each and are accepted individually; the two
N-bearing cycles count 7 and 9 and are not; so `all(accepted)` is false, the component is asked as
one circuit, and 14 electrons satisfy Huckel. `circuit_boundary` then keeps the edges lying on
exactly one *chordless* face -- and the dependent eight-ring **is** chordless, because its
would-be chord is a two-bond path through an atom it does not contain, not an edge. So each of the
nitrogen's two bonds lies on two faces, both are excluded from the perimeter, and the nitrogen is
flagged aromatic with two single bonds.

The shape in general: **two eligible rings sharing a path of two or more bonds -- bridged, not
ortho-fused -- in a component whose joint count passes while a constituent ring's does not.** The
3-8 cycle window is not the cause and narrowing it fixes nothing; for the para case both
N-bearing cycles are seven-membered.

### The fix

The atom mask is derived from the accepted circuit's own edges, in `aromaticity.py` and in both
device kernels (`metal_retro.mm`, `cuda_retro.py`), which carried the same two lines. Every branch
that marks an atom aromatic now supplies an edge for it in the same statement, so "an aromatic
atom lies on an aromatic bond" holds by construction rather than being checked at the end. It is
also what the module already said a delocalized atom is -- see `circuit_boundary`, where azulene's
transannular bond is not on the circuit.

**On the device this was silent, not fatal.** Neither kernel has a counterpart of the invariant,
so Metal wrote the flag into the atom word and then read it back two ways in the same loop:
`config_select` took the *aliphatic* configuration from the bonds for valence while the hydrogen
split took the *aromatic* branch from the word. The flag is carried into the next edit's hydrogen
law, so it did not stay cosmetic. Measured directly: on `N1c2cccc(c2)-c2ccccc21` the old shader
returned `A=AROMATIC` where the host said `ALIPHATIC`. No existing test could see it, because the
WL seed reads the total hydrogen count and the bond classes and never the aromaticity word, so the
token agreed while the molecules did not.
`test_metal_perception_agrees_with_the_host_on_a_bridged_ring_system` now compares the word itself.

### Two things deliberately not done

* **The `ValueError` is still a `ValueError`.** Making it a `ChemistryRejection` would buy
  per-application containment for the *next* defect of this family, but `CMol.__post_init__` also
  enforces dtype, shape and charge-range contracts whose failure is a programming error that must
  stay loud, and this crash is what led to three real engine fixes. Now that perception cannot
  produce the state, the invariant is reachable only by a caller handing `cmol_from_exact_arrays`
  an `A` it made up -- which is exactly the case that should crash.
* **The chemistry is still accepted.** Bridging a biaryl's meta positions with one N-H is not a
  Buchwald product anyone would get, and the application now *succeeds*, producing a well-formed
  but absurd bridged bicycle. Refusing it on strain grounds is a separate capability the engine
  does not have. Perception's job was to not build a molecule that is not one.

One tempting alternative is a trap worth recording: excluding the dependent cycle from the faces,
using the `gf2_rank` machinery already in the module, also makes the model consistent -- and gives
13 aromatic atoms over 15 bonds, a fully delocalized tricycle. That tricycle is bridged rather
than ortho-fused, so it cannot be planar and cannot be aromatic. RDKit says 12 and 12. The counts
are pinned in the test for this reason.

## Known issue: a pre-edit explicit hydrogen count outliving the edit

`Hartenfeller 15`, `16a` and `16b` on an N-H triazole each produce one malformed precursor per
outcome -- the nitrile, as `H=[3,0,0] h=[3,0,-1]` for acetonitrile. Total hydrogens right, the
explicit/implicit split impossible. Guided replay keys CMols and dies on it with
`ValueError: WL identity requires exact CMol hydrogen counts` (`identity.py:130`), taking the
whole run.

### The cause is `reaction.py:591`, not the builder

    explicit_h[:survivor_count] = np.where(
        product.A[survivor_idx] == AROMATIC,
        product.H[survivor_idx] - product.h[survivor_idx],
        0,
    )

The survivor carry writes an explicit hydrogen count for every atom that was aromatic **in the
product**, with no regard for what the right-hand side makes of it. An aromatic `[nH]` is stored
with its hydrogen *explicit* -- `H=1, h=0`, deliberately, so it writes back as `[nH]` rather than
`n` -- so the carry hands `explicit=1` to an atom the RHS turns into a nitrile nitrogen, whose
edited total is 0. `cmol_from_seed_data` then computes `total_h - explicit` as `0 - 1`.

The comment at `cmol_builder.py:461` already names this hazard: aromatic B and C store all their
hydrogen as implicit precisely "so a later survivor carry (reaction.py) cannot re-inject an
explicit H". Aromatic heteroatoms do the opposite, for the SMILES round-trip, and nothing stops
the carry there.

### Two fixes that look right and are not

**Rejecting a negative split in `cmol_from_seed_data`** discards valid chemistry. Measured: every
outcome on the N-H tautomer carries exactly one malformed precursor, so rejecting makes the rule
produce *nothing* on the tautomer it exists for -- and
`test_the_triazole_rule_only_claims_the_nh_tautomer_it_can_make` pins that it must fire there.

**Storing aromatic heteroatom hydrogen as implicit**, matching the B/C branch, breaks the SMILES
writer: `smiles.py:20` takes `H - h` as the explicit count, so `[nH]` would serialise as `n` and
pyrrole would lose its hydrogen.

Nothing here is undeterminable, which is what makes both over-corrections unnecessary:
`hydrogens_for_written_state(z=7, charge=0, sigma_degree=1, bond_order_sum=3, delocalized=False)`
returns **0** for a nitrile nitrogen. The fix is to stop carrying a count the edit invalidates,
not to refuse the molecule or to change how aromatics are stored.

### The arms disagree, and the GPU is accidentally right

`cuda_retro.py:1238` and `metal_retro.mm:1502` both compute `inferred_h = h_count -
explicit_count` and then clamp: `if (inferred_h < 0) inferred_h = 0`. For this case 0 is the
correct answer, so a device search accepts the application and keys it, the route keeps it, and
CPU replay then cannot rebuild it. That is the mechanism behind "search accepts what replay
refuses". `REJECT_HYDROGEN_COMPLETION_NEGATIVE` already exists at `cuda_retro.py:168`; once the
carry is fixed neither arm should reach the clamp, and the clamp should become a rejection so the
two can never diverge silently again.

`identity.py:130` should stay a bare `ValueError`: it fires only once the invariant has broken.
`cmol.py:95` checks `H >= h` only when both are non-negative, so `H=0, h=-1` passes the one
central validation -- worth closing so no producer can build this state at all.

## Fan-out, and the `(high retro fan-out)` marker

Reductions go in `templates-medchem.csv` whatever their fan-out, and the reason is a distinction
worth stating: **a protecting group is auxiliary, a reduction is constitutive.** Switch the
protecting groups off and the route's skeleton-building steps are all still there -- it stays
strategically correct and a chemist supplies the protection implicitly. Switch a reduction off and
nothing substitutes for it: the oxidation state has to change somewhere, so the branch simply dies
short of stock. There is no point in a file that disables a reduction.

So a broad reduction is annotated rather than separated. Any rule whose classification ends in
`(high retro fan-out)` fires far more often than it pays for, and the marker exists so the set can
be grepped when a search is spending too much.

Measured over 400 n1 reference products, against the steps each rule reproduces across 500 routes:

| rule | steps covered | applications | products touched |
| --- | --- | --- | --- |
| `MC 87` nitrile to primary amine | 14 | 8 | 8 |
| `MC 88` amide to nitrile | 6 | 41 | 40 |
| `MC 89` benzylic deoxygenation | 3 | 169 | 133 |
| `MC 90a` sulfoxide to sulfide | 0 | 11 | 11 |
| `MC 90b` azine N-oxide reduction | 2 | 196 | 128 |
| `MC 91a` intramolecular FC alkylation | 3 | 32 | 25 |
| `MC 92` intramolecular FC acylation | 1 | 11 | 8 |
| `MC 93` decarboxylation, alpha to ring N | 3 | 168 | 117 |
| `MC 94` O-demethylation | 1 | 95 | 89 |
| `MC 95` methoxyazine to ring-oxo | 2 | 14 | 14 |

`MC 93` is the one this table changed. Written as `[cH:1]>>[c:1]-COOH` it fired **2,024 times on
376 of 400 products** -- five applications a molecule, because every aromatic CH is a candidate --
and that was a chemistry error before it was a cost one: aromatic decarboxylation is not general,
it needs activation. Restricting it to a carbon alpha to a ring nitrogen, which is what both
reference examples actually were, is both more honest and twelve times cheaper. It loses one of
the two: a thienopyridine 3-acid, decarboxylating at an unactivated position, which is harder
chemistry than the rule should claim.

**Read that table carefully, because the two columns do not measure the same kind of thing.**
Coverage is agreement with the PaRoutes reference: a step counts only when a rule reproduces the
precursor set that route actually used. Applications are proposals, whatever they are. `MC 90b`
fires 196 times and covers 2 because "this azine was an N-oxide" is a sound proposal at every
aromatic nitrogen and these particular routes took it twice.

So low coverage is **not** evidence that a rule is not worth having. Solving a target does not
require matching the reference route -- any route that reaches stock counts -- so a rule can
reproduce no reference step and still be what cracks a target nothing else reaches, and coverage
measured against n1 cannot see that at all. `MC 90a` covering zero here means only that PaRoutes
never reduced a sulfoxide in these 500 routes; it fired 11 times, so it is finding them.

What the table does establish is the **cost**: `MC 89` and `MC 90b` together add close to one
application to every molecule in the corpus, paid at every node of every search. Whether that
cost is worth paying is a question about solve rate, so it was measured that way -- 539 targets
that survive the full 8x8/16x16/64x16 ladder against Enamine stock, run at 8x8 with the three
rules and without them:

| | solved | |
| --- | --- | --- |
| with `MC 89`, `MC 90a`, `MC 90b` | 88 / 539 | 16.3% |
| without them | **91 / 539** | **16.9%** |

**Removing them solves three more targets, not fewer.** One target is reachable only with them and
four only without, and the mechanism is the fan-out: 8x8 admits at most sixty-four expansions, and
an extra application on nearly every molecule displaces better candidates from a width-limited
selection. The cost is not just wall clock -- it is search that does not happen.

**And at 64x16 the sign does change, which settles it.** The four targets the three rules block at
8x8 were re-run at the effort a campaign would actually use, together with the one target that
needs them:

| 64x16, five targets | solved | mean |
| --- | --- | --- |
| with `MC 89`, `MC 90a`, `MC 90b` | **5 / 5** | 1.56 s |
| without them | 4 / 5 | 2.15 s |

All four blocked targets solve once the budget is wide enough to absorb the extra applications,
the one target that needs the rules still needs them, and the run carrying them is the faster of
the two -- a solution found earlier ends the search earlier. So the three rules stay, and the 8x8
result was a displacement artefact of a narrow budget rather than a property of the rules.

The lesson is about the instrument, not the chemistry. A fixed-width comparison at low effort
charges a broad rule for crowding the frontier while giving it no room to pay that back, so it
reports the cost and hides the benefit. The same comparison one tier up reverses. Neither tier is
wrong; a conclusion drawn from one of them without the other is. The `(high retro fan-out)` marker
stays on all three, because the cost is real at narrow effort even though it is repaid at wide.

## `typical_yield`: four bands, not two decimals

Every forward row carries a yield, and it is one of exactly four values:

| band | value | what it means | examples |
| --- | --- | --- | --- |
| robust | `0.90` | runs routinely, little to optimise | Boc on and off, acetylation, silylation, CuAAC, hydride reduction of a ketone, ester hydrolysis, amide and sulfonamide from an acyl or sulfonyl chloride |
| reliable | `0.80` | works with ordinary care | Suzuki, reductive amination, Williamson, SNAr, azide displacement, nitro reduction, alkene hydrogenation, acetal protection |
| moderate | `0.65` | substrate-sensitive, often needs a screen | Buchwald, Heck, Stille, Negishi, Sonogashira, Mitsunobu, Wittig, Grignard, thermal Huisgen, Friedel-Crafts, Vilsmeier, benzylic bromination |
| hard | `0.50` | variable, usually needs optimising | the multi-bond heterocycle condensations -- Fischer indole, Friedlaender, Niementowski, benzimidazole and its chalcogen siblings, Hantzsch thiazole, decarboxylative coupling, Sonogashira-then-cyclise |

**Bands rather than a number per row, on purpose.** A yield depends on the substrate far more than
on the reaction class, so a curated `0.83` for "Suzuki" would claim resolution nobody has. What the
bands *do* carry is an ordering, and the ordering is defensible: a Boc protection really does beat a
Fischer indole, and that ordering is the whole signal a price can use. Most of the table sits in the
two middle bands, with `reliable` the largest and the two extremes the smallest; **the median is
`0.80`**, which is the only part of that distribution anything depends on -- see the catch-all note
below -- and it is pinned by test rather than by this sentence.

**These are class-level priors and not measurements.** Nothing here was measured on these templates.
The column exists so the number a route is priced on is one a chemist can argue with -- disagreeing
with "a Fischer indole is a 0.5 step" is a conversation, where disagreeing with "worth three steps"
is not -- and so that an ML or quantum-chemistry yield predictor has somewhere to plug in, with
these as its fallback.

**The catch-all for anything unnamed must stay at or below the curated median**, and this is a
measured trap rather than a style note. `vertsy.tropical.shape_of` takes the *minimum* over an
application's templates, so a default cheaper than the curated values makes every uncurated template
systematically win -- the fault `vertsy.costs` records as "which side this knob penalises is decided
by which rules happen to be curated, and that is uncorrelated with which chemistry is right". Every
forward row is banded, so the catch-all is reached by the **retro** rules no forward row names,
which is most of them.

## `severity`: what a competitor costs you

Every forward row also carries a severity, and it is one of exactly two values:

| band | value | what it means | examples |
| --- | --- | --- | --- |
| competing | *(blank)* | the competitor makes a second product; you lose yield and pay a separation | a second amide coupling on a diamine, an alkylation at the other nitrogen, an SNAr on the other ring |
| destructive | `destructive` | the competitor consumes the substrate; no separation recovers it | N-halogenation of a free amine, aniline or amide N-H by NBS, NCS or NIS; proto- and ipso-halodeboronation of an arylboronic acid; dehydration of a primary amide by SOCl2 or POCl3 |

**One bit, because the question is one a chemist settles without a literature search**: does this
competitor cost you a separation, or your substrate? Ranking 204 rows on a finer severity scale
would claim exactly the resolution the yield bands above decline to claim, and on an axis nobody
has measured at all. Blank is the default, so the column is six edits and 202 trailing commas.

**Why this is a column and not a harder price on the competitor channel.** The competitor channel
counts distinct products a pot could make, and it was priced, judged and retired: it fires on
roughly two applications in five and is only 1.2x enriched on the applications a veto row
independently condemns, which is a near-uniform tax. The per-row table says why -- every row it is
enriched on is one `templates-vetoes.csv` already *refuses*. What that measurement could not see is
that the channel was averaging two different facts. A second product is a separation; a destroyed
substrate is not a separation at any price. Splitting them is what gives the rare, severe half a
chance to be the signal the whole channel was not, and the split needs no new price algebra,
because a channel already carries its own curve.

**A destructive row names its reagent on the left, always.** The claim such a row makes is "this
reagent is in the pot and it eats a bystander group", so the reagent has to be a left-hand
component and the row has to state a `requires` class. A row that left the reagent implicit would
match on substrate alone and say nothing about the medium -- and one requiring nothing would
compete in every pot, which is the failure mode the `requires` column exists to avoid.

**The substrate group must be a bystander, not a participant.** This is the curation trap in the
family: a Suzuki protodeboronation row would fire on 100% of Suzuki applications, because there the
boronic acid is the Suzuki's own partner rather than a casualty. That is a uniform tax on one rule
family, which is the pathology this column was introduced to escape. The signature in a run is a
row firing on nearly every application of one retro rule.

**On a destructive row, `typical_yield` is read as the degradation efficiency** -- how completely
the reagent destroys the group, not how much product it delivers. The same four bands, read upward:
`0.90` the group is gone, `0.50` a substantial fraction survives. The column keeps its vocabulary
so one parser and one band test cover both readings, and the sign is carried by `severity` rather
than by a second column. A route that uses a degradation on purpose needs to know how well it
works, and this is that number.

**`severity` is independent of `probes`, and deliberately not checked against it.** Today the two
coincide -- the six destructive rows are exactly the six with no `probes` -- but they are different
claims. `probes` is a pairing fact ("no retro rule is my inverse"); `severity` is a chemistry fact
("this consumes the substrate"), and the table already holds the counterexample: `FWD MC 88`
dehydrates a primary amide to a nitrile as somebody's *intended* reaction, with a partner, which is
the same bond change `FWD DEG 5` and `FWD DEG 6` describe as a degradation. The same reaction is a
synthesis when you asked for it and a degradation when SOCl2 does it to your amide on the way to an
acid chloride. So no test asserts the biconditional in either direction: it would forbid the
divergence the column exists to permit.

## `probes` and `scope`: inverse pairing is not reagent scope

`templates-forward.csv` carries two relations to the retro rules and they are deliberately
separate columns.

`probes` says **"I am this retro rule's inverse."** It is round-tripped:
`test_a_pairing_that_can_be_composed_does_compose` splits the row's example product with the retro
partner and requires the forward rule to rebuild it, which is the only check that catches a retro
rule emitting a chloride paired with a forward rule wanting a bromide -- a pairing that passes both
one-sided checks and then reports silent on every real application.

`scope` says **"the reagent that performs this retro rule also does what I do."** That is not an
inverse relation and cannot be round-tripped: an ester reduction is not a ketone reduction's mirror
image, it is the other thing the same hydride reaches. Overloading `probes` with it was tried first
and the round-trip test rejected it, correctly -- and the exemption list it would need,
`RETRO_PRECURSORS_DO_NOT_REACT`, is keyed by retro code alone, so exempting `FWD MC 85a` + `MC 5`
would also have exempted `FWD MC 5` + `MC 5`, which must compose. Two columns, one union in
`load_forward_library`, and both invariants survive.

### Scope is asymmetric, and getting that wrong prices clean steps as mixtures

The sites channel asks what else the intended reagent makes from this pot, so scope has to name the
reagent's reach and not merely "related chemistry":

| retro rule | assumed reagent | forward rules in scope |
| --- | --- | --- |
| `MC 5`, `MC 6` | borohydride | aldehyde and ketone reduction |
| `MC 85a`, `MC 85b`, `MC 86` | aluminium hydride | those, plus ester and amide reduction |

A borohydride leaves an ester alone, so the ester reductions must stay **out** of `MC 5` and
`MC 6`'s scope while those two stay **in** the ester rules'. The first attempt made it symmetric
and a keto-ester then reported two products under `MC 6` -- charging a mixture for a step NaBH4
does cleanly. Measured after the fix: a keto-aldehyde reports 2 under `MC 6` and a keto-ester
reports 1, while the same keto-ester reports 2 under `MC 85a`. That difference is the
chemoselectivity call, and it is the whole reason the column exists.

The assumed reagent is written into each rule's `classification`, because the pairing encodes it
and a reader of a route should not have to infer it from which forward rows happen to be listed.

## How far the forward table covers the retro rules

The forward table exists to measure selectivity, so what matters is not how many rules have a
partner but how much of the *work* is probed. Counted over the 1,000 n1 and n5 reference routes:

| | |
| --- | --- |
| retro rules with a forward partner | the clear majority |
| reference steps reproduced by a probed rule | roughly three in four |

**Deliberately not exact here.** Both figures move every time a row is added, and a number written
down in prose is a number that will be wrong without anybody noticing. Where a count has to hold,
it is pinned in a test that fails when it drifts -- `tests/vertsy/test_forward_templates.py` keeps
the pairing and arity counts. Where it is only context, as here, it is given as an order of
magnitude and the way to measure it is given instead:

* partnered rules: `load_forward_library(...).paired_rules` over the six curated files.
* reference steps reproduced: `COVERAGE_ROUTES=1000 pytest
  tests/vertsy/bench_disconnection_coverage.py::test_bench_disconnection_coverage`, a multi-hour
  sweep on the host arm.

The second figure is the one to watch, and the shape of its history is the point rather than its
value: it was *about half* while partnered rules were also about half, because the rules doing
the most work were the ones without partners -- `MC 3` nitro reduction at 254 steps, `Cpl 1` amide
coupling at 249, `MC 43a`/`MC 43b` ester hydrolysis at 256 between them. Ten forward rows for the
busiest unprobed rules moved it to roughly three in four without touching the rule count much,
which is the right way round: a forward row for a rule that never fires measures nothing.

**Unprobed is a reported state, not a defect.** A rule with no partner is counted as unprobed and
its applications are priced at nothing rather than guessed at, so the gap is visible in the
coverage line rather than silently assumed away. The 231 still unpaired are mostly halogen and
positional variants of SNAr and coupling families whose chemistry is already probed through a
sibling, plus the selenium and phosphorus sets, which no reference route exercises.

**An `r` code is an inverse, not a variant.** Closing sibling gaps mechanically -- a code joins the
row that probes its twin -- is right for `Hartenfeller 33j/k/l` and wrong for `PG 14` and `PG 14r`,
which install and remove the same acetamide. The first attempt did not distinguish them and
`test_an_adder_and_its_remover_are_inverses_on_a_polyol`'s round-trip caught it: a forward row
cannot claim both directions. Only same-direction twins may share a row.

## `templates-phosphorus.csv`

P-C bond formation and phosphonate ester hydrolysis, separate from the medchem set for the
reason the selenium rules are: the reagents are not in a general catalogue. Dialkyl phosphites
and chlorodiphenylphosphine are absent from the Enamine blocks and from the PaRoutes n1 stock
alike, so `P 2` and `P 4` terminate only where phosphorus stock is loaded too. `P 3` is the
exception and works today, because diphenylphosphine *is* in the n1 stock.

Phosphonate drugs are real medicinal chemistry -- tenofovir, alendronate, fosamprenavir -- so
this is not exotic chemistry the way selenium is. It is separate because triarylphosphine
synthesis, which the same file covers, is ligand chemistry rather than drug chemistry, and
because a rule that cannot reach stock in the default configuration does not belong in the
always-on set.

`stock-phosphorus.smi` pairs with it and is switchable with it, and every rule in the file now
terminates on it: diethyl phosphite for `P 2`, triethyl phosphite for `P 4`, diphenylphosphine
for `P 3`, and chlorodiphenylphosphine for the forward direction. So the set is portable --
`P 3` solved the phosphonic acid target before this file existed, but only because the n1 stock
happens to carry diphenylphosphine, and that is no longer what it rests on.

`P 1` is guarded with `[#6:5]-` so that it declines phosphoric acid, which is otherwise a
phosphonic acid with one more oxygen. Phosphorus coordination is written as `D4` for the
phosphoryl rules and `D3` for the phosphine ones, which is what separates `P 2` from `P 3`
without a recursive `$(...)`.

## Nitrosation, and why `MC 81` carries a warning

`MC 81` takes an N-nitrosamine back to a secondary amine and nitrous acid. The reaction is as
robust as chemistry gets and `ON=O` is already in the n1 stock, so it terminates immediately.

**N-nitrosamines are potent genotoxic carcinogens**, and the reason this rule exists is not that
anyone wants one: nitrosation of a secondary amine followed by reduction is a standard route to
a 1,1-disubstituted hydrazine, and the nitrosamine is the intermediate. A route that *ends* on
one is a different matter, and the classification string says so, so a human reading a report
sees it next to the step rather than having to recognise `N=O`.

## `Cyc 2` and the fusion guard

`Cyc 2`/`Cyc 2b` take a pyrimidine-2,4-dione fused to an aromatic ring back to an ortho-amino
arenecarboxylic acid (or its ethyl ester) plus urea -- anthranilic acid + urea to the
quinazoline-2,4-dione being the textbook case.

Both carry `[c;x3:7][c;x3:8]` on the shared bond, and the guard is load-bearing rather than
decorative: without it the rule matches uracil and thymine, whose ring is not fused, and
breaking it leaves an aromatic bond belonging to no circuit -- `OrphanDelocalizedBondError`, a
crash rather than a wrong answer. Ring *connectivity* (`x`) is used instead of ring membership
(`R2`) because the GPU pipelines run with ring perception off and would ignore the latter.

## `stock-selenium.smi`

Every row carries a supplier catalogue number, confirmed against Sigma-Aldrich and TCI
listings, because a stock file that mixes checked and unchecked availability quietly turns
"route found" into a claim nobody verified. A few rows are reagents generated in situ rather
than bottled, and those name the catalogued precursor they come from instead: `[SeH2]` and
`[Na+].[SeH-]` from Se powder, and the four selenols from the corresponding diselenide by
borohydride reduction.

There was a second, unverified file alongside this one, for organoselenium reagents that are
chemically real but were not in any supplier listing. It is gone, and the reason it could go is
worth keeping: searched as targets against this file, all but two of its rows turn out to be
compounds the rules can simply *make* -- substituted diselenides, dialkyl and cyclic selenides,
the 2-substituted selenophenes -- so dropping them costs a step or two of route length and no
solves. The two the rules cannot reach, potassium selenocyanate and 2-phenylbenzoselenazole,
are not any rule's precursor either: `templates-selenium.csv` is the only rule file that
mentions Se, and it has no selenocyanate or selenazole chemistry. An unverified stock file is
only worth its risk when something depends on it, and nothing did.

**Why the selenols are in the verified file.** `Se 3a-c`, `Se 4a-c` and `Se 5` all emit a
C-substituted selenol as a precursor, and no rule *produces* one retrosynthetically -- a selenol
is either stocked or the branch is a dead end. Alkyl selenols are foul and air-sensitive enough
that they are rarely bottled, which is why they are written here as in-situ rows: reducing the
diselenide to the selenolate is how `RSe-` is generated in practice, and all four diselenides
are catalogued. Note the pairing, which is easy to get backwards: `Se 3` and `Se 4` want an
*alkyl* selenol whichever side is aryl, because the mapped precursor keeps the aliphatic carbon;
only `Se 5` can ask for an aryl one.

**Why any of this is hand-written.** The Enamine building blocks vendored here contain
**no** selenium at all, and the PaRoutes n1 stock contains six. ZINC was the obvious source
and is useless for this: a SMARTS search for `[Se]`, and for `[#34]`, against
`ZINC20-ForSale-22Q1` returns **0 hits out of 1.64 billion purchasable compounds** -- ZINC's
element filter excludes selenium. MolPort, eMolecules and Mcule returned nothing for `[Se]`
either. PubChem PUG-REST and the Wikidata query service were down throughout. Supplier
catalogue search was the only thing that worked.

Two things the catalogues changed about the picture:

* **Selenols are barely sold; diselenides are.** `R-SeH` oxidises in air, so the commercial
  form is `R-Se-Se-R`: diphenyl, dimethyl, diethyl and dibenzyl diselenide are all stocked,
  while benzeneselenol was not in the results at all. `Se 3` and `Se 4` disconnect to a
  selenol, which is therefore an optimistic precursor. **A template cannot fix this**: the
  honest disconnection is `R-Se-R'` to `R2Se2` plus `R'X`, and a symmetric dimer needs one
  mapped atom duplicated on the right-hand side, which reaction SMARTS cannot express and
  the linter's repeated-map check would reject anyway. The diselenides are in stock so a
  route can end there; the selenol step before it is the part to read sceptically.
* **The cheapest selenium source cannot be a stock row.** Selenium powder (Sigma 229865,
  CAS 7782-49-2) is where `Na2Se` and `NaHSe` come from, but Vesma rejects a bare `[Se]`
  with `ConfigurationError: Atom 0 (Se, charge 0) occupies no declared configuration`. So
  `Se 6` produces `[SeH2]`, which sits in the unverified file annotated as generated in situ.

## Merged into `templates-medchem.csv`: coupling

**Nothing in the vendored library couples an amide from a free carboxylic acid.** Every
amide disconnection there -- Hartenfeller `47` Schotten-Baumann, iSCB `68a/b` -- returns an
acid *chloride* or bromide. Amide formation is the most-used reaction in medicinal
chemistry and is overwhelmingly run with a coupling reagent on the free acid, and carboxylic
acids outnumber acid chlorides in any catalogue by a wide margin, so this was the single
largest hole in the set. `Cpl 1` fills it; `Cpl 2` adds ester aminolysis.

`Cpl 1` carries `!@`, so it is the intermolecular partner of `Mac 5`, which is the same
disconnection restricted to a ring bond.

This is what makes **targets 5 and 9 reachable**: each is a DOTA conjugate, and one
application of `Cpl 1` returns DOTA itself, which `stock-chelators.smi` now carries.

## Merged into `templates-medchem.csv`: cycloaddition

`Cyc 1` is the azomethine ylide [3+2] cycloaddition onto an alkylidene-oxindole, which
builds a spiro[pyrrolidine-3,3'-oxindole]. It is written for a PROTAC
joining a pomalidomide-derived cereblon binder to an MDM2 binder built on exactly that
scaffold, and nothing in any other rule set can make that ring system.

Target 2 decomposes completely once `Cpl 1` and `Cyc 1` are present:

1. `Cpl 1` splits it into the PROTAC amine and the spiro-oxindole carboxylic acid.
2. `Cyc 1` opens the spiro-oxindole into glyoxylic acid, cyclohexylamine and the
   alkylidene-oxindole.
3. Hartenfeller `50` Wittig takes the alkylidene-oxindole to 5-chloroisatin and a
   2-chloro-6-fluorobenzyl halide, both catalogue items.
4. The PROTAC amine opens under Sonogashira `46` and reductive amination `30`.

The rule is deliberately narrow: the oxindole is spelled out in the pattern rather than
matching any pyrrolidine, which would fire on a large fraction of the corpus.

## `stock-chelators.smi`

Eleven macrocyclic and acyclic chelators. The 12-membered ring shared by targets 5 and 9 is
eight carbons and four nitrogens -- **cyclen**, the core of DOTA -- and both
targets are DOTA mono-amide conjugates. A route should buy that ring, not close it, which
is why these are stock rather than a macrocyclisation rule.

Structures are hand-written and checked by molecular formula: cyclen `C8H20N4`,
DOTA `C16H28N4O8`, NOTA `C12H21N3O6`, cyclam `C10H24N4`, TETA `C18H32N4O8`,
DTPA `C14H23N3O10`, EDTA `C10H16N2O8`. DOTA tris(t-Bu ester) is included because it, rather
than the free acid, is the reagent normally used for conjugation.

**Each row records whether Enamine already covers it**, checked the way the sugar file was
not: by the keyer's own identity rather than by string equality. Five of the eleven --
cyclen, TACN, cyclam, EDTA and NTA -- are already in the 988,112-row Enamine catalogue and
are kept only so the file stands alone when used without it. The other six are genuinely
absent, **DOTA among them**, which is what makes this file load-bearing for targets 5 and 9
where the sugar file was redundant. None of these carry stereocentres, so unlike the sugars
the string and keyer answers happen to agree here; the check was still run properly.

## `stock-common-reagents.smi`

Sixty-seven rows: the reagents every medicinal chemist has on the shelf -- methanol,
methylamine, ammonia, iodomethane, acetaldehyde, acetic anhydride, Boc2O, TsCl, TBSCl, TMSCN --
and the bulk building blocks a screening catalogue also omits, currently eight sugars. They are
in stock for the same reason the chelators are -- a route should buy them, not make them.

Every sugar row was taken from a Sigma-Aldrich product listing, not from recollection, and every
constitution is machine-checked: molecular formula and ring pattern, which is what distinguishes
the galactose diacetonide (a pyranose, rings 5/5/6) from the glucose one (a furanose, 5/5/5).
The absolute configurations follow the named compound and have not been independently re-derived;
a chemist should spot-check them before anything stereochemical is claimed.

**No stock file carries a CAS number.** They were dropped deliberately and a test refuses their
return. CAS Registry Numbers are the one identifier here whose rights holder has asserted claims
over redistributing its registry, and they were redundant for us three times over: the SMILES *is*
the structure, the name is what a chemist reads, and the catalogue number is how the molecule is
actually bought. Supplier names stay -- naming a supplier to identify its own product is the use a
trademark permits, and a bare catalogue number nobody can attribute is worth less than nothing.

**A sugar route's stock claim is sound up to constitution and no further.** `vesma.keyer` drops
stereochemistry, so D-glucose, D-galactose and D-mannose are one key. Everywhere else that is a
fair trade -- two stereoisomers of a drug-like scaffold are made the same way -- but on sugars the
stereochemistry *is* the chemistry. All three are genuinely purchasable, so a route ending on any
of them is not false; it may simply have bought the wrong isomer. The collision is pinned by a
test so a change in the keyer is noticed rather than discovered.

That limit is also why a near-miss is not worth shipping. A search for beta-D-galactose
pentaacetate returned the **glucose** pentaacetate instead; adding it would have told the search
that the galactose peracetate -- a direct donor for `Gly 1a`, and not confirmed purchasable -- was
in stock, because the two share a key. The free hexoses do not have this problem: galactose itself
is confirmed, so the collision invents nothing.

The fourth column separates the two: a **commodity reagent** is consumed to make a bond, a
**commodity building block** ends up in the product. Both belong here, because the test of
membership is "small, bulk, and absent from a screening catalogue" rather than what the molecule
goes on to do. `stock-selenium.smi` is a separate file because organoselenium is genuinely exotic
and only a handful of targets ever want it; a sugar is not that, and putting it behind its own
opt-in flag would mean every campaign that needs one has to know to ask.

They are here because of a measurement, not a hunch. Five of the seven hard-7 targets fail
with the search *terminating*, not running out of budget: 87 to 368 expansions against a cap
of 48 passes. The root is healthy -- 10 to 26 rules match each target, giving 21 to 37
disconnections -- so the collapse is below it. A census of every leaf in the reachable graph
says only 1-2% are dead, but names them:

    target 2:  CBr, CC#N, CC(C)N, CCl, CI, CN, CO
    target 6:  CBr, CC=O, CCl, CI, CS, N
    target 7:  CBr, CCC=O, N

Methyl bromide, methanol, methylamine, ammonia, acetaldehyde. A dead precursor kills its
whole AND-node, so every methylation and reductive amination dies on its trivial partner
however good the other half is -- which is how a tree with 27 root disconnections stops at
87 expansions.

The Enamine file is a **screening-compound** catalogue, and checked against it 11 of these 58
are already present -- benzyl alcohol, benzylamine, benzoic acid, TsOH, Boc2O -- and every one
of the 11 is benzyl-sized or larger. Nothing at C1-C3 is there. These are absent by the
catalogue's scope, not by availability, so each row carries a CAS number and is marked
`commodity reagent` rather than given a catalogue number that would be invented.

## Why these files are split the way they are

The dividing line is **whether a run can sensibly turn it off**, not what kind of chemistry it
is.

`templates-medchem.csv` is the always-on set, carrying the `MC`, `FGI`, `Cpl` and `Cyc`
prefixes. These were three separate files until the distinction stopped meaning anything --
medchem already held functional-group interconversions of its own, so a separate FGI file was
drawing a line through the middle of one idea. The prefixes are kept inside the file, so a
row's lineage survives the merge.

Three sets stay separate, each for a reason that can be checked rather than asserted.

A **template** set earns its own file when its rules cannot terminate on a general catalogue.
That is a different question from whether some *stock* is opt-in, and conflating the two is what
previously kept `ammonium` and `sulfonium` out of `medchem`.

* `selenium` -- Enamine carries no selenium at all, so these rules are unusable without
  `stock-selenium.smi` and would be pure fanout in an ordinary run. `Se 6` returns a bare
  `[SeH2]`, which only that file supplies. It is the last element here needing its own anything.

  Sulfur is the contrast that makes the rule concrete. Enamine holds over twelve thousand dialkyl
  sulfides, so `S 1` and `S 2` terminate normally, and `S 3`'s bare `[SH2]` turned out to be an
  ordinary commodity once four catalogue entries were actually looked up -- hydrogen sulfide,
  sodium hydrosulfide, thiourea and potassium thiocyanate all sit in `stock-common-reagents.smi`
  now. Nothing about sulfur was exotic; it had simply never been checked.
* `carbohydrate` -- C-glycosylation needs an anomeric leaving group and gets its stereochemistry
  from neighbouring-group participation, so written loosely it fires on any alpha-alkoxy centre.
* `protecting-groups` -- measured at 12 extra applications on a polyol sugar against 21 from
  medchem and Hartenfeller together, and 0 on ibuprofen. The cost lands exactly where protection
  is wanted, and a protecting-group-free search is a real thing to ask for.

`hartenfeller` stays separate for a different reason: it has upstream lineage, and its rows
are traceable to a published table rather than authored here.

`hartenfeller` stays separate for a different reason: it has upstream lineage, and its rows
are traceable to a published table rather than authored here.

## The ranked reaction-class checklist

A second checklist, 348 reaction classes ranked by how reliably they work. It carries names
only -- no SMARTS -- and the repository states no licence, so it is read here and every
pattern is our own, written flat. Its two filters disagree and the difference is large:
`good` spans rank_score 4 **and** 5, so `rank_statement in {good, excellent}` is 348 rows
where `rank_score in {5, 6}` is 99. This used the 348, in 58 families.

Probed family by family, with the gaps authored in batches:

| family | probed | gaps | added |
|---|---|---|---|
| functional group interconversion | 40 | 25 | 22 |
| protections + C-C bond formation | 32 | 1 | 1 |
| heterocycle formation | 35 → 16 | 3 | 8 |
| everything remaining | 45 | 8 | 17 |

Two lessons from doing it.

**Probe with substituted rings.** Heterocycle formation first looked like twelve gaps and was
mostly a bad probe: Hartenfeller's benzimidazole, benzothiazole, thiazole, indazole and
pyrazole rules all want a substituent, so the bare parent ring matched nothing. Re-probed
properly, three of the twelve were real.

**Run the lint on every row.** It caught mapped-then-dropped atoms in three consecutive
batches -- the azide nitrogen in MC 46, the CDI carbonyl in MC 60, the nitrous-acid nitrogen
in MC 61 -- all atoms that genuinely leave, all invisible to eyeballing.

Four entries were declined rather than written. Wurtz coupling gives statistical mixtures.
Decarboxylation and alkene hydrogenation read in retro as "any alkane from any acid" and
"any alkane from any alkene at any position": fanout with no chemistry in it. A
triazolopyridine rule was drafted and dropped once it became clear its [4,3-a] and [1,5-a]
isomers come from different precursors and one rule cannot tell them apart.

## Where else stock could come from

Four catalogues already on disk were surveyed for what they add *below Enamine's floor* --
molecules of 15 heavy atoms or fewer that Enamine and `stock-common-reagents.smi` both lack:

| source | unique | <=15 heavy atoms | new |
|---|---|---|---|
| PaRoutes `n1-stock` | 13,432 | 8,304 | 2,575 |
| PaRoutes `n5-stock` | 13,326 | 8,434 | 2,543 |
| SyntheMol REAL | 137,656 | 104,780 | 2,491 |
| SyntheMol WuXi | 14,974 | 11,653 | 2,514 |

PaRoutes `n1-stock` is the interesting one and needs no download -- 109 KB, already vendored,
already registered as `n1`. Being the leaf set of real USPTO routes, it carries exactly the
reagents nobody lists: `O=O`, `ClCl`, `C#N`, `[Li][CH3]`, `[CH3][Mg+]`.

**It changes nothing.** Added to hard-7 it gives the same five targets and the same expansion
counts to within a few nodes; on hard-22 it is worth one target. Those 2,575 molecules are
mostly not the ones any particular route wants, which is the argument for the hand-written
fifty-eight rather than against it: a route needs methanol specifically, and a catalogue that
happens to contain 2,500 other small molecules does not supply it.

ZINC20 remains the option if a broader reagent layer is ever wanted. Its 2D tranches at
files.docking.org/2D/ are binned by heavy atom count and logP with a four-letter code -- size,
polarity, reactivity, purchasability -- so the small in-stock bins can be taken directly, and
the file server needs no login where the catalogue pages are captcha-gated. Worth knowing
before paying for it: ZINC states its catalogues are filtered for drug-discovery relevance, so
the smallest commodity reagents may be missing there too.

## What raw68k was for, and what it found

raw68k is not trusted for routes, but it answers one question nothing else could: is the
curated set thin, or is the search broken? Run on hard-7 at 64x16 it solves **7/7**, every
target budget-limited at 16/16 passes. The curated set at the same effort solved 2/7 while
*converging* at 8 to 11 passes. A search that converges has run out of expandable nodes, so
the search was fine and the coverage was not.

Reading the disconnections raw68k used -- as leads, not as rules to copy -- named the gaps:

* **Reductive amination existed only for ketones.** Hartenfeller's rule matches `[CH1:1]`, a
  methine. `R-CH2-NH-R'`, the aldehyde product and the commonest amine disconnection in
  medicinal chemistry, had no rule at all. MC 25a/25b/26.
* **Azole nitrogens could be arylated but not alkylated.** MC 27a/b/c. The nitrogen takes
  `[a:2]`/`[a:3]` neighbours, not `[c]`: a pyrazole N1 sits next to N2.
* **Esters came only from alcohols.** MC 29a/b/c alkylate the acid with a primary halide.
* **No quinoline synthesis at all.** MC 30a/30b.

## A faster probe than the bench

Each bench run spends about twelve minutes rebuilding stock fingerprints, because the cache
is keyed on the merged stock set and any change to it re-keys all 920k molecules. For
template work a depth-limited solvability probe is enough and answers in seconds: read the
Enamine file into a set of SMILES strings, then ask whether a target is reachable from that
set within N disconnections. It skips the keyer and the fingerprints entirely. The proxy is
sound for what it is used for -- a string match and the keyer diverge only on stereochemistry,
which neither the keyer nor the RDKit arm preserves anyway.

It answers a sharper question than the bench's solve counts. After MC 29, hard-7 targets 1,
4, 6 and 7 are solvable within depth 4 and targets 2, 3 and 5 are not solvable at any depth
up to 4 -- so those three are short of rules or of depth, not of search effort, and no
increase in width or passes will reach them.

## Correction: `iSCB 64a/b/c` matched sulfone sulfur

The Williamson-like thioether rules were written `[C:1]-[S:2]-[C:3]`, and a bare `[S:2]`
matches sulfur at any oxidation state. Applied to a sulfone or a sulfoxide they broke an
S-C bond and left the sulfur to pick up a hydrogen, emitting `C[SH](=O)=O` and `C[SH]=O` --
species that do not exist. The census above found them sitting among the dead leaves.

Fixed by pinning the sulfur divalent, `[S;D2:2]`, which is the sulfide the reaction actually
needs. Verified both ways: sulfones and sulfoxides now yield no SH-bearing precursor at all,
and dimethyl sulfide is still matched by all three halide variants.

## Selenolane, and what target 15 still needs

`Se 6a-c` closes a cyclic selenide from a 1,4-dihalide and a selenide source, which is the
literature route to the selenosugar: the 2,3-protected 1,4-ditosylate or dihalide of a
pentitol plus `Na2Se`. `[SeH2]` and `[Na+].[SeH-]` are in `stock-selenium.smi` as the
selenide source.

So target 15 now has a full disconnection chain -- `Se 1` to the selenolane diol and a
sugar epoxide, then `Se 6` to a dihalo-diol and selenide -- but **whether it terminates
depends on sugar-derived building blocks that were not verifiable here**. The selenolane
diol itself, 1,4-anhydro-4-seleno-D-arabinitol, is a research intermediate rather than a
catalogue compound as far as could be checked.

## `templates-protecting-groups.csv`

Protecting groups in both directions. Removed: Boc, Cbz, Fmoc, trifluoroacetamide, acetamide and
phthalimide on nitrogen; TBS, TBDPS, TMS, THP, trityl, benzyl ether and acetate on oxygen; and
benzylidene acetals and acetonides on 1,2- and 1,3-diols. Added: every one of them, under a code
ending in `r`.

**Both directions, deliberately, and this reversed an earlier decision.** The file used to carry
twelve removers and only four adders -- the diol acetals -- on the grounds that an adder fires on
every hydroxyl and amine in every molecule where a remover fires only where the group already is.
Measured against the 21,097 PaRoutes products, the removers fire on little of the corpus:

| group | share of corpus | group | share of corpus |
| --- | --- | --- | --- |
| Boc | 6.75% | THP | 0.36% |
| TBS | 1.38% | Fmoc | 0.03% |
| Cbz | 0.92% | benzylidene 1,2 | 0.03% |
| phthalimide | 0.46% | trityl, acetonide 1,3, benzylidene 1,3 | 0.01% |

A bare `[C:1]-[OH]` adder would sit near 60%, and that number is still true. What changed is the
weight put on it against what removal-only costs:

* **A route whose last step is a deprotection cannot be built at all** without an adder, because
  nothing can propose the protected precursor. The protect/react/deprotect detour the selectivity
  price exists for ends in a deprotection whenever the target carries no protecting group of its
  own, so removal-only guarantees "the price never lengthened a route" by construction rather
  than measuring it.
* **It is the largest single gap in the collection.** On 500 n1 reference routes, 230 of the 501
  unimolecular steps the set could not reproduce were deprotections -- Boc 73, acetate 47, benzyl
  38, TBS 19, Cbz 17, THP 12, phthalimide 10, TMS 8, trifluoroacetamide 6.
* **The fanout is priced, not just incurred.** The default heuristic is `|u| x (1 - T)` -- small
  and close to stock -- so an adder's successor is scored strictly worse than the molecule it came
  from and is ranked below it. It costs a match pass and a fingerprint, not a subtree.
* **This file is optional.** A run that does not want the adders does not load it, which is the
  property that makes the trade reversible per run rather than per repository.

`test_no_protecting_group_rule_adds_a_group_going_backwards` still asserts that every code does
what its name says, so a remover cannot quietly become an adder; the list of adders is now most of
the file rather than four exceptions.

## Merged into `templates-medchem.csv`: FGI

Eighteen functional-group interconversions: an alkyl halide from an alcohol, split four
ways by substitution, and an alpha-halo ketone from a ketone, split two ways.

**Each is spelled to the substitution where the reaction actually works, not to every
halide.** That is a chemistry decision rather than a branching one. `[C:1]-[Cl,Br,I]` would
also claim an aryl chloride, which no Appel or `SOCl2` route makes, a tertiary halide, which
eliminates, and an alpha-halo ketone, which does not come from the alpha-hydroxy ketone.
The four alcohol rules therefore require a primary carbon with an sp3 neighbour, a benzylic
carbon, an allylic carbon, or a secondary carbon with two sp3 neighbours; an sp2 carbonyl
neighbour fails all four, and the alpha-halo ketone gets its own pair of rules that
disconnect to the ketone instead. Verified: `Clc1ccccc1`, `CC(=O)CCl` and `CC(C)(C)Br` all
return no match.

Branching was not the reason for the narrow spelling, and it is worth recording that the
fear was misplaced: alkyl halides are simply rare in drug-like molecules. Even the naive
`[C:1]-[Cl]` reaches only 2.64% of the 21,097 PaRoutes products, and the four narrow rules
sit at 0.57%, 0.56%, 0.01% and 0.00%. The earlier "~60%" figure in this file refers to the
*alcohol* side of an adding rule, which is a different shape entirely.

**Nothing in any rule set could previously make a plain alkyl halide** -- `CCCCCl` matched
zero rules across all 209 -- while a dozen rules *consume* one as a precursor. The
alpha-halo ketone was the same story, and Hartenfeller `7a-c` thiazole and `52` imidazole
both require one.

### Target 15, end to end

With these, both halves of the selenonium salt reach commercial material:

```
target 15
  Se 1   -> selenolane diol            +  benzylidene sugar epoxide
                |                            |
           Se 6 | -> dibromo-triol + H2Se    | PG 9  -> OCC(O)C(O)C1CO1
        FGI 1b  | -> monobromo                 29a   -> OCC(O)C(O)C(O)CCl
        FGI 4b  | -> xylitol                 FGI 1a  -> xylitol
```

**The pentitol is already in the Enamine stock**, and an earlier draft of this file claimed
otherwise. The claim came from comparing canonical SMILES as strings, which is not how the
search compares molecules: `vesma.keyer.short_ring_wl_keyer` and `cmol_rdkit_smiles_keyer`
both drop stereochemistry, so the flat `OCC(O)C(O)C(O)CO` this chain produces keys
identically to the catalogue's `OC[C@@H](O)[C@H](O)[C@@H](O)CO`. Checked across all 988,112
Enamine rows, every sugar the chain could want -- glycerol, erythritol, a pentitol, a
hexitol, glucose, ribose, xylose, arabinose -- is present as some stereoisomer. A separate
sugar stock file was written and then deleted as redundant; the lesson worth keeping is that
**stock coverage must be tested with the keyer, not with string equality.**

For the record, the three the chain depends on are commodity items: xylitol (CAS 87-99-0,
Sigma X3375), meso-erythritol (149-32-6, Sigma E7500) and D-(+)-arabitol (488-82-4,
Sigma A3381).


## Known issue: two replay bugs behind the onium rules

Routes using the onium rules are selected, counted as a solve, and then dropped at replay,
leaving `solved=True` with an empty route archive -- worse than a clean failure, because the
solve rate still counts it. `vertsy.replay.ReplayMismatch` reports why; without it a dropped
route leaves only a count.

**This section has been wrong twice, so the numbers below are the diagnostic's, not an
inference.** It first blamed the epoxide-forming `Se 1`/`S 1`, inferred from which variants
appeared in surviving routes. It then blamed selenium. Measured, there are two bugs:

| | affects | what the diagnostic says |
| --- | --- | --- |
| **A, arity** | `Se 1`, `Se 2`, `S 1`, `S 2`, **`N 2`** | recorded 1 precursor, replay emitted 2 applications of 2 |
| **B, no match** | `Se 6` | recorded 3 precursors, replay emitted 0 applications |

On the selenonium target at 256x48, 20 selected routes fail as 13 of bug B and 7 of bug A, which
is why selenium never yields a surviving route. Target 16 has only bug A and still replays 18
of 20; target 14 replays 3 of 5.

Bug A's discriminator is the **rule shape, not the element** -- every affected rule is a
charge-changing onium disconnection `[C:5]-[X+;D3:1](-[C:6])-!@...`, and ammonium shows it as
readily as selenium. The recorded child key is neither of the two fragments the CPU matcher
produces, which is the signature of the two precursors being returned **un-split as one
molecule** somewhere on the search path.

Raising the replay cap does not help and was measured: `AOSTAR_HOST_REPLAY_CAP=20` leaves
target 15 at 20 failures of 20, identical to cap 1, because the failure is that no application
matches at the node rather than that the first assignment was unlucky. It also showed that no
selection yields more than one distinct replayed route here, so the application multiplicity
guided replay allows for does not arise on these targets.

This is Vesma CPU against Vesma Metal -- replay uses `CpuReactionMatcher` and
`short_ring_wl_keyer`, and only `output_key` touches RDKit. The backends may differ on caps
and ring perception, not on the arity of a disconnection, so both are backend bugs and the
templates stay as written. Applied directly to the intermediate both backends agree, two
applications of two precursors each, so the divergence needs the search path to reproduce.

## `templates-forward.csv`

The other half of a retro template: what the reaction *does*, rather than what it came from. Written
here because a retro SMARTS cannot be reversed mechanically — a left-hand side is a match pattern and
may contain alternation, a right-hand side is a rewrite program where alternation means nothing, and
`parse_reaction_smarts` validates the right-hand side for concrete elements and bond types, which a
query-laden product side is not. Verified by `tests/vertsy/test_forward_templates.py`.

A different column layout from every other file here, because these rows are not rules a search
applies — they are measurements a search takes *about* rules it applies:

```
forward_code, forward_template, probes, example_precursors, example_product, classification
```

`probes` is a **semicolon**-separated list of retro `template_code`s, semicolons because a code
contains a space (`FGI 1a`). It may name codes in any table here or in the vendored Hartenfeller
one; every name has to resolve, and a test checks that.

### What the probe is for

`vertsy.selectivity` runs these rules on a disconnection's precursors and counts the **distinct
products**, by key. One means the step is clean. Several mean competing sites, chemoselectivity or a
competing reaction, and `vertsy.costs.selective` then prices the step as several steps.

Counting products and not match sites is the whole design. Xylitol has five hydroxyls and a
chlorinating agent matches all five, but it makes **three** distinct monochlorides: the molecule is
meso, so its two primary positions are equivalent and give one product, and of its three secondary
positions two are equivalent and give a second, leaving the central one to give a third. Keying
through the run's own keyer removes the symmetry for free. Counting sites would say five, and would
be counting a molecule's symmetry as though it were a selectivity problem.

### Three constraints that decide how a row is written

**A forward rule has the arity of its partner's precursor side.** A retro template may drop a bulk
reagent: `PG 9` removes a benzylidene acetal and its precursor side is the bare diol, with the
benzylidene carbon unmapped and deleted. The application the probe is asked about therefore has *one*
precursor, so a two-component forward rule would match nothing and read as silent. `FWD 9` invents
the aldehyde on its right-hand side instead, mirroring the convention `PG 9` already uses. Where the
retro rule does name both partners — `Hartenfeller 47` gives an acid chloride and an amine — the
forward rule is bimolecular and `vesma.cmol_union` builds the one graph a two-component pattern has
to match into.

**The right-hand side may be written Kekulé or aromatic; both work.** Measured on `FWD 9`, both
spellings produce the same aromatised product, because vesma recomputes aromaticity after the edit.
This is the *opposite* of the retro side, where 35 of 146 Hartenfeller rows were dead on arrival for
exactly this reason (see `../external/hartenfeller-2011/README.md`): a retro left-hand side has to
*match* an already-aromatic molecule, while a forward right-hand side only has to build one.

**No SSSR primitives**, the same constraint every other file here observes, because the Metal and
CUDA pipelines switch ring perception off. Bond-level `@`/`!@` is fine.

### Unprobed is not clean

A retro rule with no forward counterpart is **unprobed**, and an unprobed application is charged
nothing. This is the distinction the whole mechanism rests on: raw68k is tens of thousands of rules
with no forward counterparts, and a run over it must not come out looking selective.
`vertsy.selectivity.ProbeReport` makes coverage a number a run states rather than one a reader
assumes, and separates unpaired from *silent* — a forward rule that fires on nothing is a curation
defect, not a chemistry finding.

The lint here is a **round trip**, not mass balance. A retro library is linted for atoms going
missing, because a retro step that loses a fragment is always a defect. A forward rule legitimately
loses atoms — a condensation expels water — and legitimately gains them — an alkylation adds a whole
group — so mass balance says nothing about one. What does say something is whether the rule
reproduces its own partner's product from its own partner's starting materials, which is one
parametrized test per row.

### Flattening a ChemoDOTS rule, and the two ways it goes wrong

The vendored workbook under `../external/hartenfeller-2011/raw/` holds the forward SMIRKS the retro
table was hand-derived from, with a `RULE` column that maps onto `template_code` — so the pairing
is nearly free. What is not free is that **54 of its 62 rules are dense in recursive `$(...)`
constraints**, which vesma refuses on purpose. Flattening one is mechanical once the recipe is
known, and every row here was round-tripped before it was committed.

| recursive constraint | flat form |
| --- | --- |
| `$([#6]~[#6])` — has a carbon neighbour | write the neighbour: `[CH2:1](-[#6:2])` |
| `!$([#6]=O)` on an sp3 carbon | `[CX4]` — a carbonyl carbon is `CX3` |
| `$(C=O)` on an acyl carbon | write it: `[C:1](=[O:2])` |
| `$(S(=O)(=O)[#6])` | write it: `[S:1](=[O:2])(=[O:3])` |
| `$(O[#6])`, `$(N[#6])` | write the neighbour and map it |
| `!$(N=*)`, `!$([ND3])`, `!$([ND4])` | explicit counts: `[NH2;D1]` primary, `[NH1;D2]` secondary |
| `$(c:c)` | write the aromatic neighbour, or the whole ring |
| `;r5` — an SSSR primitive, unavailable | write the ring out: `[nH1:2]1:[a:5]:[a:6]:[a:7]:[a:8]:1` |

**Three defects the round trip caught, all of which look fine on the page.**

*A written neighbour must be mapped.* Flattening `$([#6]~[#6])` to `[CH2:1](-[#6])-[Br]` and then
writing `[CH2:1](-[#6])` on the right hand side does not carry the matched neighbour through — it
**deletes** it and creates a fresh unmapped carbon, so propyl bromide plus ethanol gave diethyl
ether. This is the same lost-atom-map defect `vertsy.template_lint` hunts in the retro library, seen
from the other direction. Map it: `[CH2:1](-[#6:2])`. Note the asymmetry with an atom the pattern
never mentions at all: an *unmatched* substituent survives untouched, so `[NH1;D2:2](-[c:3])` keeps
its N-methyl without ever naming it. Only an atom that is matched and then absent from the right
hand side is deleted.

*A hydrogen count that changes must be stated.* A forward rule gets away with a Kekulé product ring
under RDKit because the toolkit aromatises what it builds; **vesma does not dehydrogenate for you.**
A precursor's hydrogen count is *carried over* unless the right-hand side restates it, and it turns
up three times over:

| what changes | example |
| --- | --- |
| a CH2 becoming an aromatic CH | Paal-Knorr: `[cH:2]:[cH:3]` |
| a heteroatom losing its H to the ring | benzothiazole: `[sH0:2]`, benzoxazole `[oH0:2]`, benzimidazole `[nH0:4]` |
| a plain CH becoming a **ring-fusion** carbon | Fischer indole: `[cH0:4]` |

Get one wrong and the edit is refused with a negative hydrogen count, a Hückel failure, or an
orphan delocalized bond — three messages for one cause. **The safe habit for a newly created
aromatic ring is to state the count on every atom of it**, which is what the heterocycle rows do.

*An impossible assignment refuses the whole application.* The Fischer indole matched its ketone's
*methyl* as the alpha carbon as well as its CH2, and a methyl cannot become an aromatic carbon — so
the rule has to say `[CH2:3]`, which is the chemistry anyway, since Fischer indolisation needs an
enolisable position. A rule that admits an assignment the model cannot express is a rule that
sometimes raises instead of answering.

*And an atom list cannot be built.* `[#6,#7:5]` is fine on a left-hand side and refused on a right,
because a right-hand side is an edit and an ambiguous element is not a thing to construct. Where a
ring atom only needs to be carried through, `[a:5]` on both sides does it.

### Seventy-nine rows, and the arity that decides what each is good for

Ten are unimolecular and sixty-nine take two starting materials, which is not an accident: **all 62
ChemoDOTS rules are multi-component**, because a reaction that joins two things is what a
combinatorial library enumerates. The unimolecular ten are functional-group interconversions and
protections, written here rather than vendored.

That split decides what a row can do. The **competitor channel runs only multi-component rules**,
so those sixty-nine are the ones that can detect a competing reaction at all. It also decides
*where* a row can run: RDKit's `RunReactants` takes one molecule per reactant template component
and `vesma.matcher.ReactionMatcher` hands it one molecule per `(molecule, rule)` pair, so the RDKit
arm can run the ten and none of the sixty-nine. Both vesma matchers run all of them — the CPU one
into the disconnected graph `vesma.cmol_union` builds, the Metal one into the packed blocks
`vesma.metal_reaction_matcher.metal_combine` writes — so the bulk of this table lives on a
vesma-native arm, and `AOSTAR_HOST_BACKENDS=cpu` or `metal` is what makes it usable.

Measured pairings, against the rule sets a run loads:

| rule set | retro rules | paired | coverage |
| --- | --- | --- | --- |
| `hartenfeller` | 183 | 176 | 96.2% |
| `medchem` + `hartenfeller` + `protecting-groups` | 353 | 186 | 52.7% |
| any set including `raw68k` | 67,486 + | unchanged | ~0.3% |

Seven Hartenfeller rows remain unpaired, and they are the ones whose forward chemistry does not
reduce to one template: the tetrazole and Huisgen *regioisomer* pairs are written as separate rows
here on purpose, and a handful of retro variants split a single forward reaction by a substitution
pattern the forward rule does not distinguish.

The chemistry authored first is where reactions genuinely **compete for the same partners**, because
that is where the competitor channel earns its keep: an alkyl halide against an alcohol, a thiol and
an amine (`FWD H 29`, `FWD iSCB 64`, `FWD iSCB 65` — three reactions, one electrophile); an acyl
chloride or a free acid against an amine and an alcohol; a sulfonyl chloride against both; and the
1,4 against the 1,5 Huisgen triazole, which is one pair of reagents and two products unless a
copper or a ruthenium catalyst picks. The heterocycle formations came next, and they compete with
each other wherever they share a bifunctional partner: a 2-aminophenol will make a benzoxazole with
an aldehyde or with an acid, and a 2-aminothiophenol a benzothiazole from the same aldehyde.

### What the measurement found, which was not what was expected

The campaign objection was that `xylitol → chlorinate → acetalise` should have acetalised first, to
block two of the five hydroxyls. Measured with these rows, **every step of both routes is a
mixture**:

| step | distinct products | worth, at `n + 1` |
| --- | --- | --- |
| chlorinate free xylitol | 3 | 4 steps |
| acetalise xylitol | 2 | 3 steps |
| chlorinate the acetal | 2 or 3 | 3 or 4 steps |
| remove the acetal | 1 | 1 step |

So the one-step route is worth 4 and the three-step detour 7 or 8, and the penalty makes the detour
*less* attractive rather than more.

That is a correct measurement and an honest limit, not a defect in the rows. A 1,3-benzylidene acetal
on a pentitol is not regioselective either, and it blocks two of five hydroxyls and leaves three for
the chlorination that follows. The literature route works because one acetal is thermodynamically
favoured and crystallises — information about *which* product dominates, which a count of possible
products cannot contain by construction. What the probe does deliver on this target is the verdict
that neither route is clean, which is more useful than a confident wrong ordering. Making the detour
win would need a ratio, or a curated note that a protection is thermodynamically controlled, and
neither of those is a product count.

### Against reactions that demonstrably worked

Hand-built cases can only confirm what was chosen when the case was written. The table is therefore
also measured against the recorded n1 reference reactions, which needs no search and no retro
library at all: a record carries its precursors *and* its product, so whichever forward rules
reproduce that product **are** the intended reaction for that record, and the pairing normally read
from the `probes` column is supplied by the data. `make bench-forward-calibration` runs it;
`FORWARD_CALIBRATION_RECORDS` sets how many records to read.

Over the whole corpus — 31,669 usable records, 116 skipped as unparseable or over the 64-atom cap a
device batch holds:

| | |
| --- | --- |
| reproduced by some forward rule | 6,814 / 31,669 (21.5%) |
| of those, one product — the probe calls it clean | 5,648 (82.9%) |
| two products | 999 (14.7%) |
| three | 134 (2.0%) |
| four or more | 33 (0.5%) |
| at least one competing bimolecular reaction | 2,578 (37.8%) |

Two numbers to read, and they are the two ways the probe can be wrong.

**Recall is 21.5%.** Four fifths of real reactions are not expressible by 96 rules, which is what 96
rules against all of medicinal chemistry should look like. A reaction the table cannot express is
*unpaired* or *silent*: charged nothing, so harmless, but also invisible. This is the honest number
for the table against chemistry rather than against the retro library it was written beside, and it
is the argument for authoring more rows rather than for trusting the ones there are.

**The false-positive rate on `sites` is 17.1%.** These reactions worked and were reported, so a step
the probe calls a mixture is either a step the probe is wrong about or one whose selectivity the
chemist bought with something a product count cannot see — an order of addition, a temperature, one
equivalent. Either way it is the rate at which the penalty charges a step that did not deserve it,
and it cannot be estimated from examples chosen by hand. At `n + 1` a two-product call prices a clean
step as three, so the cost of being wrong is real and 17.1% is the frequency of paying it. It rose
from 16.0% as rows were added, which is the trade the table makes: a rule that can see a second site
finds more mixtures, including ones that were not.

**37.8% of reactions that worked carry a competing bimolecular reaction**, which is the independent
confirmation of why the competitors channel ships measured but not charged. It agrees with the 61% of
applications seen during search below, and a signal that fires on a third of all *successful*
chemistry cannot price it. Narrowing it needs `competes` — a favourability table, a conditions table,
or an allow-list — not more rows.

The rules doing most of the reproducing are the ones a medicinal chemistry corpus is made of:
`FWD H 29d` (Williamson aryl ether, 709), `FWD H 31` (Suzuki, 702), `FWD H 47`/`47b` (amide from an
acyl chloride, 912), `FWD H 51`/`51d`/`51g` (Buchwald-Hartwig, 1,296 between them), `FWD iSCB 70`
(ester, 395), `FWD iSCB 67` (sulfonate ester, 364), `FWD H 54` (azine substitution, 337), `FWD H 48`
(sulfonamide, 295), `FWD H 30` (reductive amination, 216). A smaller slice reproduces nothing at all,
and the `silent` counter in a run's coverage report is where that shows up.

### What the price does to a real report

Whether the table is any good and whether the *price* changes anything are different questions, and
the second one has a cheap answer that needs no second search. Route extraction is a pure function
of the graph, so one solved state can be ranked twice — once with the selectivity price, once
without — and the two reports then differ by the price and by nothing else. The bench does this on
every target it prices (`TargetResult.route_ranking`), at ~20 ms against seconds of search, and
renders both trees whenever the top route moved. That figure is the second *selection* and not the
cost of extraction, which is now timed per phase and runs at ~28% of the search it follows.

Ten n1 targets whose PaRoutes reference route has three to five reactions, on the Metal arm at
32×8 against `hartenfeller,medchem,protecting-groups`, a combined set the forward table paired
with 51.3% of at the time of the run:

| | |
| --- | --- |
| solved | 9 / 10 |
| reports re-ranked | 9 |
| top route changed | 3 |
| of those, the *molecules* changed | 2 |
| route order changed somewhere | 6 |
| routes that got **longer** | 1 (3 reactions to 4) |

Over the applications those ten searches built, `sites` probed about half and left the rest
unpaired. `competitors` probes every application, and roughly three in five admit at least one
competing bimolecular reaction.

Silence is the number to watch, and the reason it has its own section below. A correct pairing
cannot produce it, so a silent probe means either the retro rule proposes partners its reaction will
not couple or the forward rule was flattened too narrowly here. Charging it 2×
(`AOSTAR_HOST_PENALTY_SILENT_SITES`, the default) moved no reported route on these ten and probed a
few hundred more applications, so at this scale it is measured rather than decisive — and curation,
not the price, is what shrinks it.

### Silence, and how far it was driven down

`make bench-silent-pairings` runs every pairing over real reference products and asks, per
application, whether *any* of its paired forward rules rebuilds the product. It is a worklist, and
working it took the rate from **13.8% to 7.6%** across 7,430 applications:

| cause | applications | fix |
| --- | --- | --- |
| `Hartenfeller 30b`/`30d` disconnect a **ketone**; both rules wanted an aldehyde | 172 | `FWD H 30k`, `FWD H 30ck` |
| `Hartenfeller 54`/`56` write the nucleophile as a bare `[N:2]` — ammonia, anilines and secondary amines were all out of scope; and `54c`/`54d` put the ring nitrogen *para* where every rule assumed ortho | ~150 | eight rows |
| `Hartenfeller 35` disconnects `[C;H1,H2]`, so a secondary alcohol is in scope | 20 | widened |
| `Hartenfeller 15` on an N-substituted hydrazide — that triazole cannot also carry an N-H, so it is a different ring | 7 | `FWD H 15r`, `15rf` |
| `Hartenfeller 20` with an arylhydrazine | 3 | `FWD H 20r` |
| `Hartenfeller 53d`-`53f` carry an amide where `a`-`c` carry a ketone | 1 | `FWD H 53n` |

Every one is a **scope** mismatch rather than a wrong reaction, which is what the vendored table's
habit of splitting a transformation by leaving group and by substitution pattern produces: one
forward rule cannot cover a family whose members disagree about what the leaving group is.

What remains is 563 applications, of which 36 are `Hartenfeller 57`/`58` — the declared case where
the vendored retro rule emits a formamide rather than an isocyanate, and silence is the correct
verdict. The rest is one to ten applications apiece.

Two of the four top changes are one disconnection reached by two co-canonical templates, where the
price simply credits the selective one — `[c:1]-!@[NH1;D2:2]-[c:3]` giving way to the azine-specific
`[c:1]1(-[N:2])[n:3][a:4][a:5][a:6][a:7]1`. That is the per-template keying working as designed, and
it is why the flag that drops the template label is reported beside the one that keeps it.

The other two changed the chemistry, and **one of them lengthened the route** — 3 reactions to 4:

```
COc1cc2c(-c3cc4cc(Cl)cnc4[nH]3)cn(C)c2cc1OC
  unpriced, 3 steps:  Suzuki onto Clc1cnc2[nH]c(I)cc2c1   <- an azaindole bearing BOTH an I and a Cl,
                                                             so the coupling has two sites
  priced,   4 steps:  Suzuki onto Clc1cc2cccnc2[nH]1      <- one halide only
                      then chlorinate the biaryl last
```

It reordered the synthesis so the ambiguous step became unambiguous: couple a mono-halide, install
the ring chloride afterwards, and pay one extra reaction for it. Not a protect/react/deprotect detour,
but the same shape and the same argument, on a target nobody chose for it. This appeared only after
the halide scope of `FWD H 31` was widened to include chloride — before that the probe could not see
the second coupling site at all, which is the composition lint earning its place.

The second is a benzylamine:

```
NC(=O)c1ccc(Nc2ccc(CCNCc3ccccc3)cc2)nc1
  <- SNAr with an aryl chloride
    NC(=O)c1ccc(Cl)nc1
    Nc1ccc(CCNCc2ccccc2)cc1
      <- unpriced:  [CH2:1]-[NH1:2]-[CX4:3]>>[C:1]-[Cl].[N:2]-[CX4:3]   benzyl chloride
      <- priced:    [#6:4]-[CH2;D2:1]-[N;H1;D2:3]-[CX4:5]>>[#6:4]-[CH1:1]=[O].[N;H2;D1:3]-[CX4:5]
```

The precursor `NCCc1ccc(N)cc1` carries an aniline *and* a primary alkylamine, so benzylating it with
benzyl chloride alkylates both — precisely the chemoselectivity the probe was built to see. The price
replaces it with reductive amination from benzaldehyde, which discriminates.

**The counterexample that was here is gone, and how it went is the point.** An earlier run of this
same experiment had the price swap a Suzuki for a Stille — a defensible selectivity argument and a
poor practical one, which read as an argument for a curated reagent-preference column. It was
nothing of the kind. `FWD H 43`, the Stille forward rule, required `[Br,I]` while its retro partner
`iSCB 66a` emits an aryl chloride, so the probe measured **0 products** — and a silent probe was
indistinguishable from an unmeasured one, so the Stille was charged as clean and beat a Suzuki
honestly measured as a two-product mixture. It won by not being measurable. Widening the rule fixed
it; a preference column would have papered over it.

### Deeper targets, where it bites much harder

The ten above are short — one to six reactions — and a price that reorders alternatives needs
alternatives to exist. Twelve n1 targets whose reference route has **six to nine** reactions, plus the
chloro-tetraol this repo's own target-15 narrative produces, at 32x16 against the same curated sets
and the Enamine stock:

| | ten shallow targets | thirteen deep targets |
| --- | --- | --- |
| solved | 9 / 10 | 11 / 13 |
| reports re-ranked | 9 | 11 |
| top route changed | 3 | **6** |
| of those, the molecules changed | 2 | **6** |
| route order changed | 6 | **10** |
| routes that got longer | 1 | **2** |

At depth the price moves most reports, and **every** changed top is a changed synthesis rather than
a template relabelling -- which is not true on the shallow set, where a third of the changes were the
price simply crediting the more selective of two co-canonical templates. The two that lengthened are the interesting rows, and one of them is the case this whole
change started from:

```
OCC(O)C(O)C(O)CCl
  unpriced, 2 steps:  chlorinate one primary hydroxyl of free xylitol
                        <- [CH2:1](-[CX4:2])-[Cl]>>[CH2:1](-[CX4:2])-[OH]
                      xylitol from reducing the aldose
  priced,   3 steps:  reduce a ketone
                        <- O=C(CCl)C(O)C(O)CO
                      built by C-C bond formation with the chloride already in place
                        <- O=CC(O)CO  +  O=C(Cl)CCl
```

The unpriced route's first step is *exactly* the objection the campaign raised: selectively
chlorinating one primary hydroxyl of a free pentitol. Measured, that step makes three products, and
the price abandons it -- choosing a route that installs the chloride by forming a carbon-carbon bond
instead of by substituting one hydroxyl among five. Not the protect/react/deprotect answer, and a
better one: it removes the ambiguous step rather than working around it.

**And the protection detour still does not win, even now that it can be built.** No reported route in
any of these runs uses `PG 9r`-`PG 12r`. That agrees with the xylitol measurement above rather than
contradicting it: an acetal on a pentitol is not regioselective either, so the detour's own steps are
mixtures, and the price is consistent in saying so. The adders exist so that this is a measurement
instead of an artifact of the template set.

Coverage at depth, over 129,485 applications: `sites` probed 75,456 (58.3%), left 54,029 unpaired,
and **8.1%** of the probed came back silent. That last number was 15.4% before the silent-pairing
pass described under `templates-forward.csv`, and it is the same fix seen from the other end -- a
forward rule that could not match its partner's own precursors reported nothing on every application
it met. `competitors` probed everything and found a competing bimolecular reaction for 71.2% of
applications, which is the same verdict as before: too common to price.

**Against `raw68k` it is inert, and two numbers say why.** The same ten targets with `raw68k` added
to the rule set put the forward table against 174 of 67,825 compiled templates (0.3%);
all ten solve, **no reported top route moves**, and three orders change. An unprobed application is
charged nothing by design, so a high-scale run over an unpaired library is not selective — it is
unexamined, which is exactly what `forward_coverage` exists to say out loud.

The second number is why that was never going to be close. Every one of those ten routes is **one or
two reactions long**, against PaRoutes reference routes of three to five, and each target took 94
seconds against 1.2 on the curated sets. 67,825 templates find a shortcut into stock for anything,
and a two-step route has no ordering to get wrong. Selectivity pricing needs a graph deep enough for
an alternative to exist, which the curated sets give and library breadth takes away.
