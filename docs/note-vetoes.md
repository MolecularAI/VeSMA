# Vetoes: structural vetoes on an application

A retro template says a disconnection is *possible*. The selectivity channels say whether the step
it proposes is *clean*, as a cost. Neither can say the step simply will not run: that a free
boronic acid does not survive an NBS bromination, that a benzylamine and a carbamoyl chloride in one
flask acylate each other before any palladium gets a turn.

A veto says that. It is a query pattern plus a scope, and when it fires the application
disappears -- it never enters the store, never gets valued, never competes for a width slot, never
appears in a route.

The design slot is already named. `src/vertsy/filters.py:30-35`:

> Structural vetoes -- a protecting group that must be present, a catalyst poison, a
> cross-reactivity rule -- are the same shape and belong here too, once someone has the forward
> chemistry to write them.

and the line the layer draws, `filters.py:41-43`:

> A veto changes the set of routes; a score changes only their order, and a wrong score therefore
> costs a ranking rather than an answer.

So this note is not proposing a new layer. It is proposing to write the chemistry that layer was
waiting for, and to remove the one thing stopping that layer from running on a GPU.

## Why a veto and not a cost

The competitors channel was measured at `flat:1.2` over 150 targets and did not earn its place:
on the 45 route changes attributable to it, 20 were worse and 14 better. Two findings from that
measurement shape this one.

**A sub-step multiplier only reshuffles near-ties.** `flat(1.2)` is 0.2 step-equivalents against 10%
for an extra reaction, and with `ROUTE_COST_SLACK = 3` it can only re-rank routes that were already
close. Eight of the 45 changes were pure leaving-group or step-order noise. If a reagent pair is
genuinely incompatible the honest price is a refusal; if it is not, the honest price is zero. The
middle ground buys reshuffling.

**The cases where a selectivity price did real work were all the same shape** -- two electrophiles
of different order in one pot. Phenoxide or amine plus a benzyl halide; a free amine plus an acyl
chloride or chloroformate; an amine plus an α-halo carbonyl; two comparably nucleophilic anilines;
boron or a free aniline sharing a flask with something more electrophilic. That is a short list of
named incompatibilities over *reagent pairs*, not a count of how many templates happened to match.
A few dozen rows would have caught all fourteen of the channel's wins and none of its twenty losses.

Hence: veto-only, no weight column. The cost machinery exists and is the right home if a weight is
ever wanted (`vertsy.costs.StepPrice`, `vertsy.selectivity.probed`), but a veto that is worth
writing is worth a refusal, and one that is not should not be written.

## What a veto is

```csv
veto_code,stage,scope,on,sense,pattern,rationale
```

* `stage` -- `expansion`, `extraction` or `replay`: where the veto is consulted. See *Stage* below.
  A pattern veto is cheap and defaults to `expansion`; the field exists from the first row because
  it is what keeps an expensive veto out of the inner loop by declaration rather than by care.
* `scope` -- `;`-separated template codes the veto applies to, e.g. `MC 6;MC 85a`. Empty means
  every template. The parser also accepts a `tag:<name>` form so a class vocabulary can arrive later
  without a format change; no tags are defined yet.
* `on` -- `product` (the molecule being disconnected) or `precursors` (what it disconnects to).
* `sense` -- `forbid` (a match vetoes) or `require` (the absence of a match vetoes). `require` is how
  a veto says a directing group must be present.
* `pattern` -- one query SMARTS. See the expressivity limits below; they are tighter than SMARTS.

### `on: precursors` matches the union, not each precursor

One mechanism then covers both kinds of real failure:

| | case | pattern |
| --- | --- | --- |
| intramolecular | a boronic acid carried through an NBS bromination | `forbid` boron, scoped to the bromination |
| intermolecular | a phenoxide and a benzyl bromide in one pot | a *disconnected* two-component pattern |

A two-component veto pattern matched into the union of the precursors is how "these reagents
react with each other" is said, and it needs no new machinery: `vesma.cmol_union` already exists for
this, `vertsy.selectivity` already matches disconnected forward patterns into it on the host arm
(`object_combine(cmol_union)`), and the device arms already have their own combiner (`metal_combine`).

### Rejection is per application, never per precursor

`filters.py:78-79` settles this: a disconnection is a conjunction, so keeping half of one would
claim a molecule can be made from some of the pieces it needs. One veto firing on one precursor
removes the whole disconnection.

## Where it attaches

Four seams exist. The table is the reason for choosing the third.

| seam | molecules? | template ids? | arms |
| --- | --- | --- | --- |
| per-application construction (`cpu_matcher.py:751`, the two kernels) | richest context, incl. the atom mapping | dense index only | **three implementations, two in GPU source** |
| a matcher wrapper (`ReactionMatcher -> ReactionMatcher`) | native per arm | injected | must re-declare `match_keyed`, whose signature differs per arm; must honour the no-orphan invariant |
| **the expansion layer (`Match -> Match`, where `filters.py` and `measured` live)** | all arms, in each one's own representation | injected | shared code; composes with `measured` and `traced` by nesting or by one mapping |
| the forward probe (`selectivity.forward_probe`) | all arms, via `key`/`combine` | injected | produces a count, not a verdict |

The expansion layer wins on everything, once one limitation is removed -- and the limitation belongs
to `filters.Keep`, not to the layer. `filters.py:91-95`:

> **Iterability is this wrapper's one requirement of a batch** ... an object array works here and a
> device batch does not ... A filter over molecules living on a GPU has to count their atoms there
> too, which is a different implementation rather than this one with a different batch.

The next section is about why that paragraph describes a property of `Keep`'s signature rather than a
property of filtering, and why removing `Keep` removes the limitation with it.

The layer also already holds the two decisions a new wrapper would have to re-derive: grouping on
`(product, rule, application)` rather than `(product, application)`, and leaving the batch
uncompacted because `expansions.add_layer` reads the hit rows and not the batch.

Choosing this seam means **no matcher is touched at all.** "Old matchers work just fine" then holds
because they are untouched, not because they were carefully preserved.

## It is a reaction score, and there is no new type

The first draft of this note invented a `Judge` type and two adapters to reach it -- one lifting the
old per-application `Keep` into it, one folding several together. Both adapters were bridging an
invented type to one already written. `search.py:153`:

```python
type ReactionScore[M] = Callable[[MoleculeBatch[M], HitArray], Column]
```

That is the signature, and `Expansion`'s own docstring already names this use case:

> A stock distance and a forward-selectivity count are the two that exist; over-reaction and **a
> negative-condition check** are the same shape, and so is anything learned.

> A name that is **absent** is a measurement nobody took, which is not the same claim as a
> measurement that came back neutral -- and keeping those distinguishable is what lets an unprobed
> rule set cost exactly what it costs today rather than being quietly declared clean.

The second passage is the "not examined is not clear" requirement, already solved: absence of a name
means nobody ran vetoes on this arm, and a sentinel inside the column means they ran and found
nothing. No lattice is needed, because the two claims live on different axes.

### What each adapter was hiding

**Folding several vetoes together is not a function-combinator problem, it is a column.** One
named verdict column carries *which* veto vetoed each hit row, or a sentinel for none, encoded
the way `costs.py` already encodes `UNPROBED = -1` and `SILENT = 0`. Several vetoes then compose
by arriving together in one mapping, which is the composition `measured`'s docstring already calls
*"better -- than nesting"*. Naming the offender rather than returning a bare bool is also what makes
a run reportable, and reporting is the whole first use (below). Ties go to the lowest veto index,
so the column is deterministic.

**`Keep` is the thing that should go, not the thing to adapt.** `Keep[M] =
Callable[[M, Sequence[M]], bool]` is a per-application Python predicate, and that is precisely what
forces an iterable batch -- it is the root of the GPU limitation rather than an incidental detail of
it. Written as a `ReactionScore` returning a heavy-atom deficit per hit row, `mass_balanced` works on
any arm that can count atoms in its own representation, and the filter becomes a threshold on a
column. `Keep` and `filtered` are deleted rather than kept alongside.

## Two arrows: measure, then refuse

```python
measured(match, reactions={"vetoed": vetoes(...)})   # produce the verdict column
refused(match, on=("vetoed", "mass_deficit"))          # drop rows by column
```

This is the split `vertsy.selectivity` already arrived at for scores -- `probed` measures and
`StepPrice` prices -- applied to verdicts. `refused` is **one function that knows nothing about
vetoes**: it reads named columns, decides which rows survive, and gathers every column parallel to
`hits` by the same index, which is what `filtered` already does and why that axis was chosen.
Everything arm-specific lives inside the `ReactionScore`, as three small implementations of an
existing type.

### The two arrows do not commute, and that is the point

Measuring then refusing costs a verdict on every application and can say what it refused. Refusing
then measuring is cheaper and cannot. So **composition order is the choice between cheap and
reportable**, and it is worth saying out loud because the cheap order looks like an optimisation and
silently loses the report.

The first use of vetoes is the reportable order with *no refusal at all*: run the candidate rows
over a sweep, measure how often each would fire and on which rules, and change not one route. That
is how the rows get trusted before they are enforced, and it is the same order of operations that
showed `flat:1.2` was not worth shipping.

### The one law

Refusal commutes with measurement **up to gathering**: every column parallel to `hits` is restricted
by the same surviving index. Get that wrong and a score column silently refers to rows that are gone.
`gathered` is the existing one-liner that holds it.

## Running the patterns

A veto is a bare query pattern, so evaluating one is a **match-only dispatch** -- no precursor
construction, no reaction edit. Every arm already has the entry point:

| arm | entry point |
| --- | --- |
| CPU | `CpuMatcher.has_match_many(patterns, molecules)` over `gmol_from_cmol` |
| Metal | `MetalMatcher.dispatch_match_packed(...)`, packed arrays plus `(pattern_idx, mol_idx)` pairs, `collect_assignments=False` |
| CUDA | `prepare_cuda_prune_batch_from_device_pairs` + `match_fused_cuda` |
| RDKit | `RdkitFilterCatalogReactionMatcher.screen`, which already evaluates N bare SMARTS over a batch |

**No kernel is modified.** Every piece exists: the pattern packers (`_pack_atom_masks`,
`_pack_bond_matrix`, `_pack_adjacency_rows`, `cuda_stack_gmols`), the match-only dispatch, and the
`pattern_token` upload cache so a fixed veto table uploads once exactly as the reaction table
does. What is left is scheduling which pairs to dispatch, which is orchestration.

This is also why vetoes run on-device from the start rather than through a host round-trip. There
is no packed-to-molecule decoder anywhere in the codebase -- the forward packers exist, the inverse
does not, and routes reach molecules only by replaying from SMILES on the CPU. A round-trip would be
new functionality, while the device dispatch is already written and already the right shape. The
simpler-sounding option is the one that needs more code.

### Scoping is the existing per-application candidate lookup

`selectivity.forward_probe` already consults `candidates: Callable[[int], Sequence[int]]` per
application on the rule column, so the matcher is asked for a handful of `(molecule, rule)` pairs
rather than an all-vs-all pass -- *which is what makes the probe affordable at all*. A veto's
scope is the same callable with the same signature: group hits by rule, look up that rule's veto
indices, emit pairs. Nothing new is invented and the cost argument transfers unchanged.

### An arm that cannot evaluate a veto

The absent-name convention covers the easy half: an arm that runs no vetoes contributes no
`"vetoed"` column, and nothing can read that as chemistry having passed. The harder half is an arm
that runs *most* of its vetoes and cannot evaluate one -- a pattern the device matcher refuses, a
precursor over the atom cap. `selectivity` already faces this and drops groups a backend could not
build rather than scoring them, because *"leaving them in would return no hits and read as SILENT"*.
A veto column needs the same third value for the same reason, and the count of rows carrying it
has to reach a report; `ProbeReport` is the shape to copy. An arm silently skipping one veto is
indistinguishable, in the column alone, from an arm whose chemistry passed it.

## What a veto may express

Tighter than SMARTS, and the limits are not negotiable per-arm:

**Available** -- element, aromaticity, charge, explicit degree, total connectivity, valence, total
and implicit hydrogen counts, isotope, ring *connectivity* (`x`), bond order, and bond-level `@` /
`!@`.

**Unavailable** --

* recursive `$()`, refused at parse time (`parsing/__init__.py:41-44`);
* chirality;
* ring *membership* (`R`, `r`), because it depends on one spanning forest's fundamental cycles and
  the GPU backends leave it unknown. `tests/data/templates/README.md:21-24` already imposes this on
  hand-written templates and a veto is bound by it for the same reason.

**This rules out lifting a published alert set wholesale.** PAINS, Brenk and the Daylight
collections -- including the copies vendored under `tests/data/external/zbh-smarts` -- are built on
recursive SMARTS. The same applies to SMARTS-RX: its class patterns are heavily recursive, so it is
usable here as a *vocabulary of names* for a future `tag:` scope and not as patterns to evaluate.
Vetoes are hand-written and flat, which the measurement above says is the right size anyway.

## First set

Ten to twenty rows, taken from the incompatibilities the route measurement actually surfaced rather
than from a published list. No template tagging yet: explicit `scope` codes, with the `tag:` form
parsed and unused so the vocabulary can arrive without a format change.

### Three rows the alternatives report found, and what the catalogue said about them

B 32 to B 34 are the first rows found by *reading a report* rather than by thinking about the
table, and they are worth recording as a method. The first readable alternatives table -- the
glycidyl methacrylate target, retro-step 2 -- lists `MC 79`, retro alkene hydrogenation, offering
three precursors of `CC(O)CC1CO1`: two enols, `C=C(O)CC1CO1` and `CC(O)=CC1CO1`, and an oxirene,
`CC(O)CC1=CO1`. Each is reported with `sites: 1`, so the forward pattern calls each one clean
product. None of them is a species a bottle holds.

**The catalogue is what settled how to write the rows, and it overruled the obvious pattern
twice.** Screening a 100,000-row sample of the Enamine building blocks:

| pattern | condemns | what they are |
| --- | --- | --- |
| `[OX2H1]-[CX3]=[CX3]` -- any enol | **39** | every one stabilised: vinylogous acids, tetronic acids, ascorbate-like lactones, fluorinated 1,3-diones |
| sp3 required on the hydroxyl carbon only | **5** | hexafluoroacetylacetone, a dimedone-like cyclic 1,3-dione, a cyanoenol -- each stabilised from the *far* side |
| B 32 + B 33, sp3 required on both | **0** | -- |
| B 34, `[OX2]1-[CX3]=[CX3]-1` -- oxirene | **0** | -- |

So "an enol is not isolable" is false as stated, and the rows say the true thing instead:
**an enol with no stabilising group on either carbon** is the keto compound's tautomer and not a
second catalogue entry. It has to be said positively -- sp3 carbon required on both sides -- because
the subset has no recursive SMARTS for "not conjugated to a carbonyl", and the same limit is why
the oxirene row names its ring by the oxygen rather than by its size: there is no `r3`. Cyclopropene
is a real compound and is deliberately left alone.

This is the loop the report was built to close: a route's alternatives name a template, the
template's output names a species, and the catalogue says whether anyone sells it.

### Nine rows the report caught over-firing, and what was left standing

The same loop run a second time, and it inverted the diagnosis it started from. `B 13` and `B 14`
fired on the step a report *recommended* on 4 of 18 fresh PaRoutes targets, under every price shape
and with identical counts -- which reads as a coverage gap, and was written up as one: "the
halogenation rules have nothing acceptable to offer on those targets". Reading the products said
otherwise. Sandmeyer (`MC 9a/b/c`) and POCl3 (`MC 82a/b/c`) had been in the library all along, and
of 12 condemnations over those targets **8 were pyridones or a halogen on a fused thiophene**.

A pyridone is not an azine. Its ring nitrogen carries a hydrogen or a substituent and *donates*:
the ring is activated like a phenol, and 2-pyridone brominates at C-3 with NBS. Two corrections,
applied to all nine rows of the family -- halogenation, nitration and formylation share the
argument and shared the flaw:

* **`[nX2]`, a divalent nitrogen.** This is the whole fix for the pyridones, and it is the thing
  that separates an azine from its oxo tautomer.
* **Meta and para as six-membered rings, not paths.** A path of aromatic bonds crosses a ring
  fusion, which is how `B 14` came to condemn chlorinating the electron-rich thiophene of a
  thienopyridine three bonds from the azine nitrogen; and meta or para means nothing in a
  five-ring, whose electron-rich positions halogenate readily. The **ortho** rows stay paths on
  purpose: a carbon alpha to a pyridine-type nitrogen is electron-poor in a five-ring as much as in
  a six, so C-2 of a thiazole is as unreachable as C-2 of a pyridine.

The all-carbon ring positions then spare 5-halopyrimidine and 4-halopyridazine, which is right
rather than a loss: those are the positions a diazine genuinely *does* halogenate at --
5-bromopyrimidine and 5-bromouracil are made that way -- while every diazine position alpha to a
nitrogen is still caught by the ortho row. Verified over 26 cases; host and device screens agree on
all 34 rows over 13 molecules, so the ring closures behave the same on the Metal arm.

**What the corrected rows left standing was two real library gaps**, and this is the part worth
copying as a method: the residue of a fixed veto is a worklist.

| gap | what the report showed | what was added |
| --- | --- | --- |
| no directed metalation **at all** | `Fc1ncc(CN2CCOCC2)cc1I` condemned correctly -- C-3 iodination of a 2-fluoropyridine does not happen, you metalate and quench | `MC 97a/b/c`, ortho to a ring halogen, with forward partners; `tag:carbanion` carries the protic-quench rows to them |
| POCl3 only **alpha** | a 4-chloro thienopyridine reachable only by chlorinating a deactivated ring, while 4-chloroquinoline from the 4-quinolone is textbook | `MC 98a/b`, gamma chlorodehydroxylation, written as rings for the lesson above |

**Measured, and it is the argument for curation over tuning.** Same 20 targets, same 18 reporting,
same 54 routes, counting *steps* a report recommends that a row objects to:

| | before | after |
| --- | --- | --- |
| condemned steps on the top route | 10 | **2** |
| condemned steps on any reported route | 33 | **8** |
| targets with any | 9 | **3** |

And the three price shapes converged: `flat:1.2`, `graded:0.1` and `flat:3` all land on the same
two steps, where before they ranged over 9 to 12 code hits. **The price shape stopped mattering
once the table was right**, which is the clearest statement available that the lever here is
curation and the multiplier only has to stay above the noise.

Three defects in the *reporting* engine surfaced on the way, each worth more than the row it
blocked: a veto that fired with no row able to name it (the measurement key is
`(precursors, rule)` and a product-side verdict can honestly disagree between two products, so the
pair is dropped -- the disputed rows are now named `B 13?`, since they are the only candidates); a
metric that counted codes where it meant steps, and so read one condemned step naming two
candidates as two steps; and a report that could end a run by asking RDKit to kekulise a product no
replay had to stand behind.

## Two axes, two kinds of veto

`Expansion` insists it carries *"exactly **two id spaces** and no third"* -- one value per precursor,
one value per hit row -- and those are exactly the two claims a veto can make:

| claim | axis | type |
| --- | --- | --- |
| these reagents will not react as intended | per application | `ReactionScore` |
| this molecule cannot exist | per precursor | `MoleculeScore` |

Refusal on the molecule axis **induces** refusal on the reaction axis through the hit rows' precursor
column, and by the conjunction rule killing one precursor kills the whole application. So `refused`
reads both axes and the induced map is well defined; there is no third concept to invent.

A veto, as specified above, is the first row. The second is what the bridged biaryl needs.

### Vetoes see one step, and that is the design

A veto says *this step will not run*. It does not say *this molecule is a bad idea*, and it should
not: a molecule nothing can make is discovered by the search failing to make it, which is what
dead-ending is for, and what the ranker and the selectivity channels are for. "This intermediate
will not survive three steps later" is not a veto's sentence, and the absence of a seam for it is
not a gap.

### What the bridged biaryl actually needs, and why it is absurd

Number biphenyl's rings A (1'-6') and B (1-6), joined 1-1'.

* **Carbazole** is an NH bridging C2-C2', ortho to ortho'. Its new five-ring shares **one** bond with
  each benzene, which is *fusion*: everything stays coplanar, the nitrogen sits at ordinary angles,
  the result is flat and aromatic and real.
* **The meta case** is an NH bridging C2 and C3'. Its new six-ring shares **two adjacent bonds** with
  ring A, which is *bridging*, not fusion: the path from C3' back to C1' outside ring A is N-C2-C1, a
  **three-atom strap across benzene's 1,3-positions**.

C1'...C3' is 2.42 A. A three-atom strap must span that, and two of its three atoms are locked inside
a rigid flat hexagon with 120 degree exocyclic angles. A [3]metacyclophane is impossible with a
flexible trimethylene strap; here the strap *is* an aromatic ring and cannot bend at all. Measured:
RDKit's ETKDGv3 finds **no 3D structure whatsoever** for the meta and para bridges over twenty
attempts at 2000 iterations, while carbazole and acridane embed immediately and come out at 2.2 and
2.0 kcal/mol of MMFF strain per heavy atom -- *below* plain biphenyl's 3.3.

So the graph signature is crisp: **fusion shares one bond, bridging shares two or more**, and a small
ring sharing two or more bonds with an aromatic ring is the absurdity. That is a molecule-axis
predicate, not a reaction-axis one.

Two ways to say it, and they are not exclusive:

* **As a pattern**, close to expressible with the bond-level `@` / `!@` technique
  `tests/data/templates/README.md` already documents for ring-size floors -- and bounded by the same
  caveat, that the technique measures reachability rather than ring size and so over-rejects on fused
  systems.
* **As embeddability.** "No 3D structure satisfies these distances" is a *general* impossibility
  predicate that catches every absurd bridged system with nobody writing a pattern for any of them.
  Far too slow per precursor per layer, and it needs RDKit, which is an optional extra here. But as
  an offline screen over a stock or template library, or as a last check on an extracted route, it is
  cheap and decisive.

## A veto creates a demand; a template supplies it

A `require` veto and a protecting-group template look like two ways of saying one thing. They are
not alternatives, they are different *kinds* of arrow, and they compose:

* a veto **subtracts** from the set of routes -- "T needs an ortho directing group" is a statement
  about T;
* a template **adds** to it -- "here is how to install a carbamate" is a statement about a reaction.

So the veto states the requirement, the template supplies it, and the search finds
`install-P -> T -> remove-P`. Omit the veto and the search runs T without P and returns a route that
does not work. Omit the template and the search cannot reach T on a substrate lacking P, which is
honest rather than broken.

`forbid` has the identical structure, which is what makes this one idea rather than two cases:

| veto | the demand it creates | what supplies it |
| --- | --- | --- |
| `forbid` a free hydroxyl under these conditions | mask it | a protection template |
| `require` an ortho directing group | install it | an installation template |

So `require` is not a special case needing its own justification. It is the dual of `forbid`, and
both are useful only *because* generators exist to satisfy them.

### This is why protecting groups exist

Protection has never won a route here, and the reason was understood: the heuristic rewards getting
smaller, protect-react-deprotect is three steps that grow the molecule first, and a ranker with no
reason to prefer it is right to refuse it. **Vetoes are that reason.** Without a veto, protection is
pure overhead; with one, it is the only way through.

That is a prediction, and it is the second thing to measure after the fire counts: adding `forbid`
rows should *increase* protecting-group usage in extracted routes. If it does not, either the vetoes
or the protection templates are wrong, and the measurement says which rather than leaving it to taste.

## Cost: what a veto costs, and therefore where it may run

**Retired: `stage`.** The field used to take `expansion` / `extraction` / `replay`, and two things
were wrong with it. It named a *mechanism*, so only someone who knew the pipeline could fill it in.
And one of its three values could not work: **extraction holds interned ids and no molecules**, so a
veto that screens a molecule -- which is every veto there is, and a conformer veto doubly -- had
nothing to screen there. The note below used to say extraction "needs no refusal mechanism at all"
and priced a veto there at infinity; that was a real mechanism for a stage that could not evaluate
the predicate in the first place.

What replaces it asks a question a chemist can answer:

| cost | what it is | may run at expansion? |
| --- | --- | --- |
| `pattern` | a subgraph match | yes |
| `geometry` | embeds a structure -- *no 3D arrangement satisfies these distances* | **no** |

That prohibition is the point of declaring a cost at all. It is a safety property enforced by
whatever schedules the veto, not a convention anyone has to remember.

`cost` and `force` are orthogonal and both needed: cost says what evaluating the predicate costs and
so where it may run, force says what a fired row *does*. The same predicate, at the same price, can
be a wall or a surcharge.

### Why not a price of infinity, even where one would work

Two reasons, and the second is the one that decides it.

The polarity is a debugging hazard: cost and utility flip the absorbing element, so "is it zero or
infinity here" has to be remembered rather than read.

**And an infinite price cites nothing.** A chemist asking why a route is missing wants *"step 3
refused by B 1 -- boron does not survive NBS"*, which is the `rationale` column read back. "Utility
zero" teaches them nothing and is indistinguishable from an arithmetic bug. Refusal is the better
mechanism precisely because it is reportable.

### The expensive case: validate at replay, cache per application

Replay is where molecules exist again -- `template_replay_routes` rehydrates them to re-run the
templates -- so that is where an expensive veto runs. The loop is: extract the best route, replay
it, evaluate the expensive rows, and if one fires, take the next route instead.

**What makes that principled rather than a swamp is caching the application's verdict, not the
route's.** A rejected *route* is one candidate gone, the verdict is not reusable, and two routes
sharing the bad step each pay for it. A rejected *application* is one value written into the verdict
column that already exists: every route using that disconnection is excluded with no
route-exclusion mechanism at all, and the loop converges because one expensive evaluation prunes
many candidates. It is memoised lazy evaluation of an expensive predicate over a shared graph, and
it needs no new concept.

The anytime schedule is already the right shape for it: the store is append-only and the search runs
several passes, so pass *n+1* sees what pass *n* learned and stops expanding what has been refused.

What a run has to report is **how many candidate routes validation refused, and by which rows**. If
that number is large while routes are missing, the next thing to try is obvious -- turn validation
off, get all the routes back, and read the veto list -- and that is the whole reason the number is
worth printing.

#### Built, and measured: the refusal replaces a route rather than subtracting one

`select_solved_routes(validate=...)` asks each candidate in report order and is told the reason it
cannot be made, or nothing. The refusal is asked **before** the diversity floor, so a route nobody
can make never becomes the thing the next route has to differ from, and the candidate that takes
its place is still chosen for differing from the routes a reader will actually see.

Two targets of the six a blind judge said the selectivity price made worse, at
`AOSTAR_HOST_VETOES=validate` with both channels at `flat:1.2`: three routes requested, three
routes returned on both, `routes_lost` **0**, and `routes_replaced` 1 and 4 -- with the reason
named per route. Before this, those same refusals dropped the route at the readable-route stage and
the report came back short.

**And on one target it fixed the chemistry.** `C=C(C)C(=O)OC(C)CC1CO1` reported, as its best
route, a Grignard onto an epoxide-bearing halide -- `CC=O + ClCC1CO1 -> CC(O)CC1CO1` under
Hartenfeller 45a, which row B 30 condemns -- because the `sites` probe reports it as one clean
product. With the replay layer refusing, the four routes built on that step are refused and route 1
becomes `MC 62`, the mCPBA epoxidation of the homoallylic alcohol, which is the answer a chemist
would write. The esterification it displaced is still in the report, at **-0.021**: cheaper, and
reported second, which is the honest statement that the order is not monotone in cost.

The verdict is not yet cached back into the store, so the convergence argument above is still
only an argument. What is built is the loop it describes.

## Closed

* **Tie-breaking between the two axes: there is nothing to break.** A refusal is a refusal, so the
  axes combine by disjunction and no winner has to be chosen. An earlier draft of this note argued the
  verdict column should name the offending veto *because that is what makes a run reportable*, and
  that was wrong: reportability comes from aggregates -- "this row fired 412 times, mostly under these
  rules" -- which need no per-row attribution at all. So the column is a plain sentinel, the counts
  live in a report on `ProbeReport`'s shape, and "why was *this* disconnection refused" is answered by
  re-running the vetoes on that one application, which is what replay is for.
* **The adjacency recompute was a slow helper, not a design constraint.**
  `_adjacency_rows_from_bonds` spelled "set bit `j`" as a multiply by `1 << j` followed by a
  reduction, which made numpy materialise an `(N, 64, 64)` **uint64** array -- 1.6 GB at fifty
  thousand molecules, to produce 25 MB. As a `packbits` it is bit-identical and 13 to 27 times faster
  (649 ms to 47 ms at N = 50k; 5.3 s to 196 ms at N = 200k). At 47 ms a layer, matching vetoes
  against every precursor is affordable and in-scope-only scheduling is an optimisation rather than a
  precondition.

## Wired: `AOSTAR_HOST_VETOES`

Three settings, and the middle one is the point. `off` is the default, and the only setting that
leaves every number comparable with the runs recorded under `docs/paper2/benchmarks/`. `measure`
computes the verdict column and acts on nothing. `refuse` enforces it.

The dependency injection is the whole of it: the table is read once beside the forward library,
its scopes resolved against the rules *this* run compiled -- the same codes-meet-ids problem
`load_forward_library` has -- and handed to whichever arms were asked for as a `VetoLayer`. An arm
supplies exactly two things for itself, a `Screen` over its own molecules and a `Combine` to build a
pot, and `layered_match` stacks the measurement onto the match and the enforcement onto that:

    reactions = {**probes, "vetoed": vetoed(table, in_scope, screen=..., combine=...)}
    match = measured(lifted, reactions=reactions)
    match = refused(match, applications=("vetoed",))   # only when refusing

So a veto arrived as **one more key in a mapping** and no new layer in the search. That is what the
reaction-score argument bought: both measurements are `ReactionScore` columns, so they compose by
being named rather than by being nested, and the arm-specific halves stay inside the scores.

The device screen is `metal_screen`, and it needed no kernel change -- a veto is a plain subgraph
question, so it is `dispatch_match_packed` at `max_assignments=1` over the arrays the batch already
holds, with the patterns uploaded once under a stable token and no molecule ever unpacked. It lives
in `vesma.metal_reaction_matcher` beside `metal_combine`, for the same reason: the injected
operation is declared where the algebra is and implemented where the packed layout is. Host and
device are asserted to agree on every shipped row, in
`test_the_device_screen_agrees_with_the_host_on_the_shipped_veto_table`.

The `rdkit` arm has no screen and **raises** rather than measuring nothing, which is the resolution
of "an arm that cannot evaluate a veto" above at the coarsest grain: the arm cannot evaluate *any*
of them, and a run that quietly reported a clear verdict on all six would be the exact confusion the
sentinels exist to prevent.

### What `measure` mode buys, precisely

`VetoReport` is `ProbeReport`'s shape with one addition: `fired`, a count **per row**. A row is the
unit of the decision -- a table is curated one row at a time -- and a pass total cannot say which
row is carrying it, nor which two rows catch the same thing. The per-row counts deliberately **do
not sum** to `condemned`, because two rows condemning one application is exactly the overlap worth
knowing about before one of them is deleted as redundant.

The claim that makes the mode worth running is that it predicts the refusing run exactly: the same
verdict is computed either way, and only whether anything reads it differs. Asserted in
`test_what_a_measuring_pass_condemns_is_what_a_refusing_pass_removes`.

First reading, two PaRoutes targets at 16x4 over medchem + hartenfeller + protecting-groups: **B 1
condemned 3 of 699 applications on the first target and nothing fired on the second.** Refusing
those three removed three of 210 unique routes. Five of the six rows have yet to fire on anything,
which is the number the sweep has to settle.

## Open, and orthogonal to the implementation

* **Which vetoes to write.** A chemistry question the framework does not depend on, and answered by
  measuring fire counts over a sweep before anything refuses anything -- which `measure` mode now
  does.
* **Which rows the competitor channel already catches.** Four of the six name an incompatibility
  that is now, in principle, visible to `competitor_probe`: B 6 is an acyl chloride acylating the
  amine before palladium turns over, and `FWD H 47` *is* that reaction, bimolecular and in the pot
  by construction; B 2, B 3 and B 4 are hydride chemoselectivity, and the reductions became
  bimolecular with `[AlH4-]`/`[BH4-]`, so the hydride is now a precursor a competing rule can
  reach. B 2's rationale still says such a step is *"invisible to the competitor channel because
  the step is unimolecular and so has no second reagent to compete with"*, and that sentence is now
  false.

  **Overlap is not redundancy**, though, and the difference is force: a veto refuses, a competitor
  charges. The blind route-pair judgement found the selectivity price re-ranks near-ties and little
  else, so the two are not substitutes. The sequence that follows is: measure soft, enforce hard --
  the channel is the discovery tool that says which collisions actually occur, and a veto row is the
  decision that a particular one is fatal rather than expensive.
