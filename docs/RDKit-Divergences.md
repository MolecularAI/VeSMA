# Comparing VeSMA with RDKit and RDChiral

VeSMA, RDKit, and RDChiral are complementary parts of the open cheminformatics
ecosystem. RDKit provides broad molecular and reaction functionality. RDChiral builds on
RDKit with retrosynthesis-oriented stereochemical checks and product handling. VeSMA uses
a smaller, explicitly declared chemistry model designed to give the same result on CPU,
CUDA, and Metal without localizing delocalized bonds.

This document explains where those models can produce different answers and why. It is not
a scorecard, and a difference does not by itself identify an error in any toolkit. Some
differences follow from supported-language boundaries, some from distinct but internally
consistent reaction semantics, and some from information that one model deliberately does
not represent.

The examples are intended to make each class concrete. Current frequencies belong in the
audit output rather than in this document, because they depend on the chemistry version,
template collection, and molecule corpus. See [Measuring current frequencies](#measuring-current-frequencies).

## VeSMA's relevant model commitments

VeSMA uses a **closed-shell, graph-topological delocalization model**.

- Every supported atom occupies one declared configuration. A configuration fixes its
  sigma capacity, graph valence, and possible contribution to a delocalized circuit.
- A delocalized bond has no hidden single or double order. VeSMA does not construct a
  Kekulé form during perception or reaction application.
- Delocalization is categorical. The model does not represent partial aromatic character,
  resonance energy, ring current, planarity, or orbital orientation.
- Unsupported, ambiguous, or internally inconsistent states produce a typed refusal rather
  than a guessed configuration.

These commitments make the model small and portable, but they are not the only reasonable
way to represent molecular chemistry.

## Summary of divergence classes

| Class | VeSMA behavior | Comparison behavior |
|---|---|---|
| [Circuit-relative delocalization](#circuit-relative-delocalization) | Counts electrons for the candidate circuit | RDKit uses its aromaticity model; RDChiral normally follows RDKit |
| [An edit releases a delocalized bond from its circuit](#an-edit-releases-a-delocalized-bond-from-its-circuit) | Refuses unless the released bond is given a concrete order | RDKit and RDChiral can retain a localized bond-order assignment |
| [A possible circuit lies outside automatic discovery](#a-possible-circuit-lies-outside-automatic-discovery) | Requests explicit delocalized annotation | RDKit answers using its own aromaticity model |
| [The RHS is a rewrite program](#the-rhs-is-a-rewrite-program) | Applies state changes and recomputes derived properties | RDKit and RDChiral do not use those derived fields as postconditions either |
| [A disconnected component has no RHS atom](#a-disconnected-component-has-no-rhs-atom) | Omits the undescribed component and reports it | RDKit removes the branch associated with the deleted mapped atom |
| [A multi-component RHS meets a ring](#a-multi-component-rhs-meets-a-ring) | Applies edits to one atom-conserving graph | RDKit follows RHS component templates; RDChiral often stitches duplicated components |
| [Hydrogen after a local edit](#hydrogen-after-a-local-edit) | Preserves the atom's closed-shell configuration | RDKit and RDChiral may retain an open-shell atom |
| [Stereochemistry](#stereochemistry) | Not represented | RDChiral validates stereochemical consistency |
| [Planarity and orbital orientation](#planarity-and-orbital-orientation) | Not represented | Other aromaticity models may encode additional heuristics |
| [RDKit sanitization and exchange](#rdkit-sanitization-and-exchange) | Keeps a native categorical-delocalized graph | RDKit may be unable to sanitize or serialize an equivalent molecule |
| [Radicals, and what a bracketed atom means](#radicals-and-what-a-bracketed-atom-means) | Reads a bracket atom as OpenSMILES specifies, then refuses an open-shell species | RDKit and RDChiral read it the same way and build the radical |

# Chemistry-model differences

## Circuit-relative delocalization

VeSMA asks whether a particular circuit has a supported closed-shell configuration and a
Hückel-compatible electron count. Its circuit selector uses element, charge, heavy-atom
degree, total hydrogen, and the topology of concrete multiple bonds leaving the eligible
ring system. It deliberately does not use a localized alternating-bond assignment inside
the circuit.

This can classify a ring differently from RDKit. A useful example is the levamisole core:

```text
pattern   [NX3][CX3]=[CX3]
molecule  S1C2=N[C@H](c3ccccc3)CN2C=C1

RDKit     retains a concrete C=C in the thiazoline ring, so the enamine pattern matches
VeSMA     represents that five-membered circuit with delocalized bonds, so `=` does not match
```

VeSMA's count for the five-membered circuit is six: sulfur and the bridgehead amine
nitrogen contribute lone pairs, the two atoms of the endocyclic C=C contribute one electron
each, and the carbon whose C=N bond leaves the eligible ring system contributes zero.
RDKit's aromaticity model reaches a different bond classification. Both answers are
consistent with their respective graph models; the comparison does not establish a
geometry-dependent physical aromaticity measurement.

Boroxine illustrates the same boundary from another direction. For a B3O3 ring, VeSMA's
configuration table assigns two electrons to each participating oxygen lone pair and zero
to each three-coordinate boron with an empty perpendicular orbital. The formal circuit
count is therefore six. RDKit does not automatically aromatize this ring unless its own
donor rules find the required evidence for participation.

This is a representation choice, not a claim that boroxine has benzene-like aromaticity.
Experimental and computational measures describe boroxine aromaticity as absent, weak, or
criterion-dependent: a [photoelectron study](https://doi.org/10.1016/j.cplett.2007.04.028)
found no significant aromatic character, while a more recent
[computational comparison](https://doi.org/10.1002/ejic.202300641) reported small values
relative to benzene. VeSMA represents only the categorical circuit result and does not
attempt to reproduce those degrees of aromatic character.

## An edit releases a delocalized bond from its circuit

A categorical delocalized bond has no concrete order to recover after its circuit is
destroyed. VeSMA therefore refuses an application that leaves an authored `:` bond outside
an accepted circuit:

```text
template  [cH:1]:[cH:2]>>[CH2:1]-[CH2:2]
molecule  c1ccccc1

VeSMA     refused: configuration_incompatible
RDKit     C1=CCCC=C1
RDChiral  C1=CCCC=C1
```

RDKit and RDChiral have a localized bond-order assignment available and can carry those
orders across the edit. VeSMA intentionally does not have one to carry. A template can make
the VeSMA outcome well-defined by assigning concrete orders to every bond released from the
destroyed circuit.

The same rule applies when the released bond lies in context that the RHS does not rewrite:

```text
template  [c:6]-[c:4]1:[n:3]:[c:1]:[c:2]:[s:5]:1
          >>O=[C:1]-[CH2:2].[NH2:3]-[C:4](=[S:5])-[c:6]
molecule  c1ccc2sc(-c3ccccc3)nc2c1

VeSMA     refused: configuration_incompatible
RDKit     NC(=S)c1ccccc1 + O=C1C=CC=CC1
RDChiral  NC(=S)c1ccccc1 + O=C1C=CC=CC1
```

These examples are reported as `configuration_incompatible` because configuration
selection encounters the released delocalized state before the later circuit-membership
check. Other edits of the same kind can reach that later check and report
`delocalized_edge_not_cyclic`; both names describe the stage at which the inconsistency was
first established.

Many such templates are meaningful only on a fused system. A surviving aromatic bond can
remain part of a second circuit in naphthalene but become an orphan when the same local
pattern matches an isolated benzene. This is best understood as a template-domain boundary,
not as a statement that one of the input molecules is chemically invalid.

The second ring has to be a *circuit*, not merely a ring, and that is a narrower domain
than it first appears. Phthalimide is the case that shows the difference: its benzene is
fused to a five-membered imide, but VeSMA perceives only the benzene as delocalized, so a
template that destroys the benzene while keeping a surviving `:` bond leaves that bond with
nothing to belong to. A benzene fused to a non-aromatic partner behaves, for this rule,
exactly like an isolated one.

## A possible circuit lies outside automatic discovery

Automatic perception enumerates simple cycles of three through eight atoms. A separate
cycle-space coverage check detects when eligible atoms contain an independent circuit that
was not examined. In that case VeSMA asks for explicit annotation rather than returning an
answer that could depend on whether the input used localized or delocalized notation.

```text
C1=CC=CC=CC=CC=CC=CC=C1    localized 14-membered conjugated ring
  -> refused: requires_delocalized_annotation

c1ccccccccccccc1            explicitly delocalized 14-membered circuit
  -> the authored circuit is checked at its represented size
```

The example is about model coverage. A 14-electron circuit is Hückel-eligible, but its
physical aromatic character can also depend on conformation and geometry, which VeSMA does
not represent and which are consequential for the
[14-membered annulene system](https://doi.org/10.1002/anie.200454188). An explicit delocalized spelling is
therefore an assertion for VeSMA to validate, not a prediction of a particular experimental
aromaticity measure.

Saturated macrocycles do not trigger this refusal: their atoms cannot form an eligible
delocalized circuit. Nor does a large ring containing an isolated alkene whose remaining
atoms interrupt conjugation.

# Reaction-semantics differences

## The RHS is a rewrite program

VeSMA interprets a reaction template's right-hand side as a construction program, not as a
second query to solve after construction.

The operative RHS fields are:

- atom existence;
- element and isotope, which surviving mapped atoms must preserve;
- formal charge;
- total hydrogen `H`; and
- bond kind.

Derived SMARTS properties such as `A`/`a`, `D`, `X`, `v`, `R`, `r`, and `x` are recomputed
from the result. Their RHS spelling is retained as neutral template telemetry, but a
mismatch does not coerce or reject the precursor. Implicit-hydrogen query notation `h` is
also not a state-changing instruction; an authored total-hydrogen count `H` is.

For example, `[c:1]>>[C:1]` changes neither bonding, charge, nor hydrogen. On benzene, the
result remains aromatic after perception. If aliphatic carbon is chemically required, the
template must author the bond, hydrogen, or charge changes that destroy the circuit.

Atom case still helps resolve an omitted RHS bond during compilation. That is construction
syntax, distinct from treating the resulting atom's aromaticity as a postcondition.

## A disconnected component has no RHS atom

The RHS also identifies which resulting material is described as precursor output. If a
mapped LHS atom is deleted and that deletion disconnects a component containing no RHS atom,
VeSMA omits the component and reports it as undescribed.

This matters for an under-specified ester template:

```text
[C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]
  >>[O;D1;H0:1]=[C:2]-[OH;D1;+0:3]
```

Atom 4 distinguishes an ester oxygen from an acid OH, but its map is absent from the RHS.
If it matches an interior carbon of an alcohol chain, deleting it can split off material
that the RHS never identifies. Returning that material as a precursor can create a
plausible but unintended compound. VeSMA therefore keeps only components anchored by an RHS
atom and records what it omitted so the template can be reviewed.

RDKit follows its product-template construction semantics and removes material reachable
only through the deleted mapped atom. The two implementations arrive at similar omissions
through different construction models.

## A multi-component RHS meets a ring

A dot-separated RHS normally describes separate precursor molecules. If the matched path
closes through bonds outside the template, however, applying an intermolecular
disconnection template to a ring is under-specified.

```text
template  [C:2]-[CH2:1]-[CH2:3]-[C:4]
          >>Br-[CH2:1]-[C:2].Br-[CH2:3]-[C:4]
molecule  C1CCCCC1

VeSMA     BrCCCCCCBr              one atom-conserving ring-opened graph
RDKit     CCCCBr + CCCCBr         two RHS product-template graphs
RDChiral  BrCCCCCCBr              duplicated mapped fragments stitched together
```

RDKit follows the two product templates declared on the RHS and copies the environment
around each mapped anchor. Because the anchors are also connected through the rest of the
ring, atoms can appear in both products. VeSMA instead edits the original graph and returns
its connected components, so breaking one ring bond opens the ring without splitting the
graph. RDChiral detects duplicated atom maps and often stitches the components back
together.

The template was generally extracted from an intermolecular transformation, while the ring
match corresponds to a different intramolecular situation. The three results are best read
as different resolutions of an application outside the template's intended domain.

**A template can now say which domain it is in**, and for one of the two spellings that
closes this divergence rather than merely narrowing it. Zero-level grouping is enforced on
the RHS as a postcondition: `>>(B).(C)` claims two precursors, `>>(B.C)` claims one molecule
in two fragments, and an application whose components disagree is refused.

RDKit reads the same notation and reaches the same place by a different route. Its own issue
tracker has this exact case -- [rdkit#6136](https://github.com/rdkit/rdkit/issues/6136),
closed as expected behaviour -- where
`[C:2]-[C:1]-[O:4]-[C:3]-[C:5]>>O-[C:1]-[C:2].[C:5]-[C:3]-O` on `C1CCOCC1` returns two
propanols instead of the diol, and the maintainer's answer is "you have asked for two
products, so you get two products", with parentheses as the remedy. Measured here on that
example:

```text
template  [C:2]-[C:1]-[O:4]-[C:3]-[C:5]>>...
molecule  C1CCOCC1

                    VeSMA           RDKit
>>B.C               OCCCCCO         CCCO + CCCO     the #6136 complaint
>>(B.C)             OCCCCCO         OCCCCCO         agreement
>>(B).(C)           refused         CCCO + CCCO     not a constraint RDKit has
```

"Refused" there is this *application* and not the rule. `(B).(C)` is how a template says
"two precursors", and it delivers them: the same rule on an acyclic ether, `CCCOCCC`, cuts the
C-O bond in two and returns `CCCO + CCCO` on both engines. What it refuses is the ring, where
the two halves stay joined through the rest of the cycle and the edit produced one molecule, so
the claim is false for that application alone. The mirror holds for `(B.C)`, which returns the
ring-opened diol and refuses the acyclic cut. The three spellings are a complete set:

```text
                    C1CCOCC1 (ring)      CCCOCCC (acyclic)
>>B.C               OCCCCCO              CCCO + CCCO        both readings: the default
>>(B).(C)           refused              CCCO + CCCO        a bimolecular disconnection
>>(B.C)             OCCCCCO              refused             a ring opening
```

So `(B.C)` is the spelling on which the two engines agree, and it is the spelling RDKit's
maintainer recommends for it. `(B).(C)` is where they part, and the reason is documented
upstream: the RDKit Book lists "component level grouping requiring matches in different
components, i.e. `(C).(C)`" among the SMARTS features it does *not* support, so RDKit reads
those parentheses as nothing more than the product-template split a dot already gives. VeSMA
reads them as an assertion and refuses the ring. Most of the library still uses the dotted
spelling, with the behaviour described above:
`vertsy.template_lint.ungrouped_components` reports the 373 shipped rules not yet migrated.

The left-hand side divides the same way. The RDKit Book's own intramolecular example,
`([C:1]=[C;H2].[C:2]=[C;H2])>>[*:1]=[*:2]`, is a ring-closing metathesis written as one
reactant template, and both engines return `C1=CCCCC1` from `C=CCCCCC=C`. But RDKit enforces
the molecularity *structurally* -- `RunReactants` takes one molecule per reactant template --
while VeSMA matches into a single combined pot and enforces it as a query constraint. That is
why the dotted form diverges in the other direction: `[C:1]=[C;H2].[C:2]=[C;H2]>>` is two
templates to RDKit, which therefore cannot be asked about one molecule at all, while VeSMA
matches it inside the diene and returns the ring. Grouping is what lets a VeSMA template say
which of the two it meant.

## Hydrogen after a local edit

When a surviving mapped atom changes bonding but not charge and the RHS omits `H`, VeSMA
preserves the atom's pre-edit closed-shell configuration. It chooses the unique hydrogen
count that completes that same configuration under the edited bonding. No global valence or
Kekulé solve is involved.

The under-specified ester template above sometimes makes this difference visible. For
example, apply it to
`O=C(CCCCC1CCCCC1)N1C(=O)OC[C@@H]1Cc1ccccc1`. Removing a C-O bond can leave carbon one
bond short of its prior configuration:

```text
VeSMA     ...N(CCc1ccccc1)...       carbon completed to CH2
RDKit     ...N([CH]Cc1ccccc1)...    open-shell [CH] retained
RDChiral  ...N([CH]Cc1ccccc1)...    follows the RDKit-built state
```

Both behaviors follow their respective state models. VeSMA is closed-shell by declaration,
so it cannot preserve the radical form. Hydrogen completion is reported as neutral
application telemetry so callers can distinguish an explicitly authored `H` change from a
configuration-preserving completion.

## RDKit sanitization and exchange

RDKit sanitizes reaction products and can decline an outcome when no acceptable Kekulé form
or valence assignment is available. RDChiral may process the same intermediate differently,
and VeSMA may have rejected the categorical-delocalization inconsistency before attempting
to serialize anything. The audit records RDKit discards separately from a backend that
simply produced no outcome.

Conversely, VeSMA can represent a categorical-delocalized precursor for which RDKit cannot
construct a sanitized molecule or canonical SMILES. Such a precursor retains a VeSMA-native
identity in the audit and is marked unkeyable in the shared RDKit-SMILES comparison space.
This is an interoperability boundary, not evidence that either native graph was silently
discarded.

A concrete instance, met in a route report rather than in the audit. `MC 82b` writes a
chloroazine back to its ring-oxo tautomer, `[c:1](-[Cl]):[n;H0:2]>>[c:1](=[O]):[nH:2]`, which is
the ordinary way to write a 2-pyridone. Applied to `Cc1cccc2nc(Cl)c(I)c(=O)n12`, whose ring
*already* carries an exocyclic oxo, it produces
`[CH3]-c1[cH][cH][cH]c2n1c(=O)c(-I)c(=O)[nH]2`: a six-membered ring with two exocyclic
carbonyls, one N-H and one bridging nitrogen. VeSMA perceives a circuit there and writes the
molecule; RDKit finds no Kekulé form and declines. The species is a real compound — the 2,4-dione,
and POCl3 really does chlorinate C-2 of a 2,4-dioxopyrimidine — but it is a dione rather than an
aromatic ring, and drawing it needs explicit bonds. Neither backend is the ground truth here, and
the template is not at fault: writing one exocyclic oxo on an aromatic ring is valid, and writing
a second in the same ring is where the two models part.

**What this means for a consumer.** Anything that turns graphs into SMILES has to tolerate it. The
alternatives report asks a canonical-SMILES keyer to name every disconnection of every molecule on
a route, so it meets this boundary routinely — and must lose the one name rather than the run
(`vertsy.replay.step_precursor_names`). A route whose own precursor lands here is a different
matter: replay cannot write it, so it is reported as a replay failure, which is the honest
outcome.

# Out-of-scope information

## Radicals, and what a bracketed atom means

Every atom must occupy a declared closed-shell configuration, so a species with an unpaired
electron is refused rather than represented. RDKit builds them without complaint, so this
shows up as a divergence wherever a corpus contains one.

TEMPO is the case worth stating, because it is a catalogued reagent and because vesma used
to get it wrong. `CC1(C)CCCC(C)(C)N1[O]` gives its oxygen one bond and, per OpenSMILES 3.1,
no hydrogen — a nitroxide radical, which vesma now refuses by name. Until the SMILES reading
was separated from the SMARTS one, vesma read `[O]` as an unconstrained query, derived a
hydrogen for it, and returned **TEMPO-H, the hydroxylamine**: a different, closed-shell
molecule, with no refusal and no telemetry. A refusal that names its cause is the point of
the model; a silently wrong molecule is what it exists to prevent.

## Stereochemistry

VeSMA does not represent tetrahedral or bond stereochemistry. RDKit's `RunReactants` applies
the template without RDChiral's retrosynthesis-specific stereochemical validation, while
RDChiral checks chirality and cis/trans consistency and can suppress an incompatible
outcome:

```text
template  [c:1]-[CH:2]=[CH:3]-[c:4]
          >>[c:1]-[C:2]#[C:3]-[c:4]
molecule  c1ccccc1/C=C\c1ccccc1

VeSMA     diphenylacetylene
RDKit     diphenylacetylene
RDChiral  no outcome after stereochemical validation
```

This is a useful capability difference rather than a disagreement about the resulting
constitution. The audit comparison deliberately strips stereochemistry from its shared
molecular key, so it measures whether RDChiral suppresses an outcome, not any
stereochemical distinctions among outcomes that survive.

## Planarity and orbital orientation

VeSMA's circuit model sees connectivity, element, charge, hydrogen, and bond topology. It
does not see three-dimensional geometry or whether nominal p orbitals overlap sufficiently
to support physical delocalization.

Consequently, a VeSMA circuit classification should be read as a result in its declared
graph model, not as a prediction of ring current or aromatic stabilization energy. Where a
different toolkit classifies a borderline fused heterocycle differently, geometry and
orbital orientation may be relevant information that VeSMA cannot adjudicate. Adding a
special graph pattern for one molecule would hide that limitation rather than extend the
model generally.

---

# Deliberate divergences from the SMILES specification

Where vesma knowingly reads or writes something other than what OpenSMILES 1.0 prescribes.
Every entry below is *loud*: it refuses rather than quietly answering something else.

## Charges and hydrogen counts the atom masks cannot hold exactly

OpenSMILES requires a parser to accept charges from -15 to +15, and permits up to nine
hydrogens on a bracket atom. Vesma's per-atom masks are buckets: charge saturates at -3
and +4, counts at 7. A molecule states its properties exactly, so rather than encode
`[C+5]` as a carbon of charge +4 -- which is what the bucket would do, and what it did
until this was checked -- the reading refuses anything outside those ranges and names the
limit.

This is a limit of the representation, not a claim about chemistry. It binds only the
molecule reading; a *query* may still use the saturating buckets, which is what they are
for, since `[C+4]` as a pattern legitimately means "+4 or more".

## An isotope written as zero is refused

OpenSMILES keeps `[0S]` distinct from `[S]` (section 3.1, Isotopes) so that a writer can
round-trip its own output: the isotope field is optional, so "written as 0" has to be
distinguishable from "not written". The specification does not claim the value is physical
-- it says outright that "there is no requirement that the isotope is a genuine isotope of
the element", allowing `[36Cl]`.

There is no nuclide of mass zero. Rather than key `[0S]` as natural abundance and silently
agree with `[S]`, vesma refuses the spelling and says why. Isotopes outside 1-999 are
refused on the same grounds; the specification's stated range is 0 to 999, and tools that
write larger values are using the field to smuggle in attachment points or group
identifiers, which is not something this model represents.

## Hydrogen atoms written as atoms are not yet supported

OpenSMILES section 3.6 permits `[H]` as an ordinary atom, and requires that an atom's
hydrogen count be the sum of its written count and its attached hydrogen atoms -- so
`[H][CH2][H]` is methane and `[2H][CH2]C` is deuteroethane. Vesma does not implement this.

These parse correctly -- `[H]`, `[2H]` and `[H+]` name the hydrogen atom, as they do for
RDKit -- and the refusal comes from the chemistry: hydrogen occupies no declared
closed-shell configuration, because the model carries it as a count on its neighbour rather
than as a graph node. `[H]`, `[H+]`, `[2H]`, `[3H]` and `[H][H]` all refuse with
`undeclared_configuration`.

Supporting them is therefore a change to the declared chemistry, not to the parser. It
needs hydrogen given a configuration and, separately, the section 3.6 summing rule that
folds an attached `[H]` into its neighbour's count -- without which `[CH4]` and `[H][CH2][H]`
would be one molecule with two identities, which the model's spelling-invariance commitment
does not allow.

An earlier revision of this document said the refusal came from the parser. That was true
when it was written: the literal scanner tried the `D H h R r v X x` primitives before
two-letter element symbols, so the `H` of `[H]` read as a hydrogen *count* and the atom
named no element. The same ordering made `[Hg]` and `[Xe]` fail to parse at all, and made
`[Rn]` read as ring-membership plus aromatic nitrogen -- radon as ammonia. A two-letter
element symbol now wins against the primitive that shares its first letter, matching RDKit
in both languages, and the periodic table runs to oganesson so that no symbol falls off the
end and parses as something else.

---

# Measuring current frequencies

The current comparison is produced by the executable audit rather than maintained as prose:

```bash
# Quick review over the first 100 frontier molecules
make AUDIT_SIZES=100 bench-precursor-audit

# Full n1-d2-f0 frontier sweep
make AUDIT_SIZES=2881 bench-precursor-audit
```

The target writes a gzipped pytest-benchmark JSON file under
`output/benchmarks/precursor-audit/cpu/`. Timing is incidental; this is a correctness audit.
It compares the sets of precursor outcomes for confirmed positive molecule-template pairs
and retains every divergence and refusal for inspection.

The paper reports a compact snapshot derived from the completed full audit: pairwise exact
agreement and the frequencies of the broad semantic classes described above. Those small
aggregates live in the tracked paper source so its main text and SI use one set of values.
The application-level JSON remains the evidence used to inspect individual cases; it is not
a build-time dependency and does not require a separate manifest or provenance file.

Useful fields include:

- `agreement_records`: pairwise exact-agreement and comparable-pair rates;
- `backend_records`: accepted, refused, unkeyable, and failed application counts;
- `divergences`: every differing molecule-template pair and its three outcome sets;
- `refusals`: every VeSMA typed refusal;
- `rhs_annotation_mismatches`, `undescribed_components`, and `hydrogen_completion`: neutral
  template and edit telemetry;
- `rdkit_discards` and `unkeyable_precursors`: outcomes outside the shared comparison space.

The ordinary benchmark record also carries its source commit and toolkit versions. Git
history and the tracked publication source retain the reported snapshot; no parallel
history is maintained in this guide.

Agreement rates are conditional on the selected molecule corpus, template collection,
positive-pair list, identity codec, and toolkit versions. They are useful estimates of what
a user may encounter when changing reaction engines, but they are not universal error rates
and do not make any backend a chemical ground truth.
