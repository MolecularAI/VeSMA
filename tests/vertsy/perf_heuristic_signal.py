"""Does anything cheap predict how many reactions a molecule is from the stock?

The search's heuristic is a guess, and until now nobody had checked how good a guess. This asks
directly, against labels that cost nothing: :func:`vertsy.interop.aizynth_depth` returns the subtree
depth of *any* node, so every interior molecule of every PaRoutes reference route comes labelled
with its distance to that route's leaves — 44,270 molecules for n1, and the label-0 class is exactly
the stock file.

Not a benchmark; a notebook that happens to be a test. It prints a table and pins nothing.

**Before believing anything in this table, rank leaves the way the search ranks them.** This file
scores a *population* of molecules against a depth label. The search scores *leaves* by
``context + inside``, where the context charges a leaf's open siblings -- a different object, and
the two have now disagreed twice. Graph edit distance won here and lost there. Tversky beta=0 won
here by +0.037 rho and was worth 0.002 of mean rank there, which is nothing, after several turns of
being recommended on this table's evidence alone. Only ``atoms from stock`` has won both, and
only it changed the live search.

The check is cheap: build a one-level state with :func:`vertsy.expansions.expansions_of`, fold it,
rank through :func:`vertsy.aostar.global_leaves`, and ask where a known-good precursor lands.
A hundred archive steps take under a minute. Run it before wiring anything in.

**The labels are upper bounds.** They come from one recorded patent route each, not from a minimum
over all routes, so a molecule reachable in two steps by some route nobody patented is labelled by
the route that was. That biases every candidate identically, which makes this fair for *ranking*
candidates and unfit for setting an absolute scale — the reason `vertsy.aostar` sweeps its step cost
rather than fitting it.

Two views of the same relationship, because they answer different questions:

- **deciles of the estimate**, restricted to non-stock molecules. Roughly a third of the labelled
  set is stock at exactly 0, and left in it collapses the bottom quantiles and distorts every one
  above. This view says whether the relationship is concave, linear or convex — which decides
  whether a one-parameter reshaping can express it.
- **mean estimate per true label**, the inverse. Calibration needs the estimate as a function of
  depth, and with a marginal this lopsided the two directions are not interchangeable.

**The Tversky family has an endpoint, and beta=0 is it.** Sweeping alpha with beta = 1 - alpha
is monotone on open leaves -- +0.469 at alpha=1 (beta=0), +0.451, +0.432 (Dice), +0.387, and
**-0.024** at alpha=0 (beta=1). So the incumbent can be improved by switching to beta=0 and by
nothing else in this family; there is no better setting to hunt for. The beta=1 end fails for a
reason worth keeping: it asks how much of the *stock* molecule is present, which every trivially
small catalogue entry satisfies, so the maximum is pinned at 1 for almost everything.

**"Is a whole building block already present" does not work either**, at least not through folded
bits: taking the largest stock molecule whose bits are contained in the query scores **-0.037**.
The idea is sound chemically -- a molecule eight steps out still contains its building blocks --
but bitwise containment of a folded fingerprint is dominated by small molecules that are
"contained" in everything, so it measures size, not substructure. It would need real substructure
matching, which is not a per-node cost the search can pay.

Taken together with the combination result below, the similarity family is exhausted. Every member
saturates by about three steps -- the incumbent moves 0.42 to 0.60 across labels one to six and is
flat from three on -- because "how similar is this to something buyable" is a local question and
distance to stock is a compositional one. A signal that scales with depth has to count what must
be *built*: bonds to form to reach buyable fragments, or a value function fitted to these labels.

**Tanimoto to a greedy union of stock molecules does not scale either**, which kills the obvious
way to borrow the combinatorial-library trick. Measured on a stratified 4.6k sample of n1 against
the same labels: rho +0.674 at k=1, then +0.665, +0.643, +0.637 as pieces are added. Unlike
containment it does not collapse -- charging the union's own unmatched bits makes it
self-limiting, and the union Tanimoto plateaus at 0.550, 0.633, 0.647, 0.649 -- but it limits at
k roughly 2 and ranks no better. The mechanism: greedy takes the single best-covering molecule
first, which is a large one resembling the whole query, and that spends the whole budget of
tolerable foreign bits. A real decomposition uses several medium pieces, none individually the
best match, and greedy cannot find them.

That is also why the Enamine REAL factorization does not lift over. There the *reaction scheme*
supplies the slots -- the product is known to be synthon A plus synthon B by a known reaction, so
each slot is optimised independently and the OR-decomposition is exact. In retrosynthesis the
slots are the unknown; finding them is the problem, so the beam has nothing to beam over.

**One-step lookahead makes the root worse, at either fold.** Scoring each root application by its
own best child -- min over rules of the fold over precursors, the Bellman form -- moves the
correct disconnection on a worked target from 13th of 39 to 30th (max fold) or 24th (sum fold),
and shrinks the spread. `min` is optimistic: nearly every molecule has some rule yielding a
stock-like fragment, so every application earns the same credit and discrimination is destroyed.

**What did help was the AND fold, with no lookahead at all.** Scoring an application by the *sum*
over its precursors rather than the max moves the same disconnection from 13th to 7th and widens
the spread from sd 0.040 to 0.171. Max cannot distinguish an application that leaves two large
open precursors from one that leaves a large one plus a stocked reagent -- both score the larger
one alone -- and that distinction is most of what a first disconnection is chosen on. Note this is
about *leaf selection*, not route valuation: the max fold is what makes a route's score its
longest linear sequence rather than its reaction count, which is the convergent chemistry one
wants, and the two folds need not be the same function.

**The sum fold does not reach the search, and that is the more useful half of the result.** The
leaf-selection finding above was measured offline on one node; wired into the bench as a fold
switch it changes nothing. Target 13 of a 22-target campaign, at three effort shapes, bottleneck
against closure: 256x48 expands 11805 either way and *sees* 47125 against 47124 molecules; 16x48
expands 753 and sees 4625 against 4631; 8nx48 expands 9401 and sees 40282 against 40286. Identical
expansion counts and a handful of molecules' difference in 47 thousand, at every width including
widths narrow enough that the top-k cut should bite.

The reason is the selector, not the fold. `committed_application_leaves` commits to the target's
best live application when the target is first expanded and reserves room for its branches on every
later pass, so the first disconnection is chosen once and then pursued -- the fold reorders leaves
*within* a commitment rather than changing which branch is taken. That is precisely the failure
mode the selector exists to prevent, and it also makes the search insensitive to sibling
aggregation. An offline ranking experiment on one node measures something the search does not use.
The switch was written, measured, and removed; closure_fold is also 4x slower, being the lazy fold
rather than the array one.

**A reachability level loses to similarity, which retires the whole A\*-relaxation idea.** The
argument for it was structural: every measure above is a *state* measure -- how does this molecule
look next to the catalogue -- and a state measure's range cannot grow with distance, whereas the
least number of retro steps within which a molecule reaches stock is a step count by construction.
Measured that way on 300 molecules stratified over labels 1 to 5, with the curated library and a
cap of three steps: rho +0.331 against +0.538 for Tversky beta=0 on the same sample.

The cap is not the reason. Restricted to the 146 molecules the search reached inside it, the level
scores +0.460 against Tversky's +0.563; restricted to labels 1 to 3, +0.351 against +0.469. It
loses on every subset. The uncensored subset also disposes of the obvious confound -- that a level
of "unreached" might mean the curated library cannot do a given USPTO reaction rather than that the
molecule is far from stock -- because every molecule in it *was* reached within three steps.

Two further reasons not to revisit it as written. It cost 1.7 s per molecule, so even as an
expensive re-ranker over the ~80 precursors of one node's applications it is over two minutes per
target. And it is **not deterministic**: bounding the recursion by wall clock makes the level a
function of machine load, and two runs of the same sample gave +0.344 and +0.318. A heuristic whose
value depends on how busy the host is cannot be shipped, whatever it correlates with.

**The selector is not swallowing the heuristic, which was worth checking and is now measured.**
The flat results above invite the theory that `committed_application_leaves` saturates each batch
so the heuristic's global ranking never acts: it asks each committed branch for the whole batch
(`descent_leaves(precursor, ..., limit=limit)`) and returns early once full, and it replaces a
commitment only when that commitment becomes *unreachable*. Instrumenting the split says
otherwise. On the unsolved campaign target `fill` -- which is `global_leaves` -- is called on
16 of 16 passes at 64x16 and supplies 923 of 925 expanded molecules; at 256x48, 48 of 48 passes
and 11,803 of 11,805; on a solved control, 16 of 16 and 911 of 911.

The commitment reserves about two leaves per run. `global_leaves`' own docstring gives the reason:
`descent_leaves` "can only offer what one solution graph exposes -- often a single molecule,
whatever ``limit`` says". The `limit` is an upper bound, not a quantity returned.

So the heuristic drives essentially all of selection, and the eight flat results above are flat on
their merits rather than through a closed channel. That makes the heuristic the real lever and
removes the selector as an explanation -- while leaving the fold result genuinely unexplained,
since a fold change should perturb the context term `global_leaves` ranks on and measurably did
not.

**A third multiplier for precursor quality (QED): measured, and it hurts.** The idea was that the
estimate weighs *how far* a molecule is and *how big*, but says nothing about whether what is left
is a reasonable molecule or an awkward one. QED is the cheap stand-in. Both directions lose:

    atoms from stock (Tanimoto)    rho(open) +0.518    the incumbent
    atoms from stock / QED                   +0.438
    atoms from stock x (1-QED)               +0.365
    QED alone                                -0.119

QED carries no depth signal on its own -- its mean label runs 0.609, 0.646, 0.617, 0.598, 0.578,
0.567, 0.552 across true depths 0 to 6, which is flat and slightly backwards. That is not a
surprise once stated: QED scores *drug-likeness*, a property of a target, and the molecules this
estimate must rank low are building blocks -- small, simple, reagent-like, and exactly the things
QED scores badly. Multiplying by it therefore penalises the molecules nearest the catalogue, which
is the opposite of what the estimate is for.

**Synthetic accessibility, tried next, points the right way and still adds nothing.**

    atoms from stock (Tanimoto)         rho(open) +0.518    the incumbent
    atoms from stock x (1+(SA-1)/8)               +0.518
    atoms from stock x (1+(SA-1)/24)              +0.518
    atoms from stock x SA                         +0.499
    atoms from stock x (SA-1)                     +0.473
    SA alone                                      +0.210

Unlike QED, SA is not backwards: alone it scores +0.210 with a monotone mean label, 1.67 rising to
2.54 across deciles. And it is **not** a restatement of size -- Spearman +0.104 against heavy-atom
count over 800 molecules, where QED runs +0.364. So it is genuinely independent information about
the right thing, which is why it was worth trying properly rather than dismissing with QED.

It still does not combine. A bare product *hurts*, and the reason is scale: SA spans 1 to 10 while
the distance spans 0 to 1, so the product is dominated by SA and stops ranking by distance at all.
Coupling it gently enough to stay a distance -- at most a doubling -- recovers exactly the
incumbent's +0.518 at two different strengths, which is the signature of a term contributing
nothing rather than of a strength not yet tuned.

The scale objection has a proper fix, and it was tried: normalise both to ``[0, 1]`` as costs and
take a **weighted geometric mean**, ``(d**x * s**y) ** (1/(x+y))``, with the exponents saying how
far each is trusted. ``sa / 10`` is the cost direction and floors at 0.1 rather than 0, which
matters -- a geometric mean with a zero factor is zero, so the natural ``(sa - 1) / 9`` would score
an easy molecule at 0 however far from the catalogue it sits.

    weight d:SA      1:1     3:1     7:1    15:1    31:1     pure d
    rho(open)     +0.471  +0.503  +0.512  +0.516  +0.517   **+0.518**

**Monotone toward the incumbent, with no interior optimum.** Every weighting is a strict
interpolation between pure distance and something worse, and the sequence is still climbing at
31:1. There is no amount of SA that helps; there is only an amount that stops hurting.

The confirming detail is that **direction does not matter**: ``x (SA/10)**0.5`` scores +0.514 and
``x ((10-SA)/10)**0.5`` -- the exact opposite claim about which molecules are expensive -- scores
+0.515. A term carrying usable signal cannot perform the same inverted. Whatever SA knows on its
own at +0.210, it is already implied by size and distance together.

So the scale mismatch was real but was not the whole story: fixing it cleanly still leaves nothing
to gain. The wider lesson stands anyway, because it is the same shape as the selectivity penalty
losing to the step cost in `docs/note-forward-selectivity.md` -- but here the honest conclusion is
narrower and stronger. This particular signal is redundant, not merely mis-scaled.

**Graph edit distance: better on the population, not better on the decision, and far too slow.**
MCS-based GED -- Bunke's `|A| + |B| - 2|MCS|` over a Tanimoto shortlist -- is the only candidate
here whose value does not saturate with depth: on 400 labelled molecules it scores rho +0.492
against +0.456, and its mean per true label runs 4.8, 7.3, 10.1, 12.8, 13.6 where the incumbent
runs 0.42, 0.48, 0.53, 0.57, 0.59. MCS timeouts are 0.0%, and widening the shortlist from 40 to
200 slightly *lowers* the gain, so the prefilter is not hiding a larger win.

It still fails where it matters. The search ranks sibling applications at one node, not a
population, and there the result is split: at the root of the unsolved campaign target GED ranks
the correct disconnection 9th of 39 against the incumbent's 13th, but across 60 known-good steps
from a route archive the incumbent is clearly ahead -- mean normalised rank 0.197 against 0.277,
top quartile 70% against 58%. The archive steps favour the incumbent by construction, since those
routes were found under it; the root node does not, but n=1 is not evidence. Net: no demonstrated
win on decisions.

Cost settles it regardless. Scoring one node's applications took **36 s** measured over those 60
steps -- not the 5 s a back-of-envelope suggested, because MCS is superlinear in molecule size and
real nodes carry more applications than the estimate assumed. Against a 60 s search that is 60%
overhead at a single node, for a gain that is unproven on decisions and, on the population, the
same +0.03 to +0.05 that Tversky beta=0 supplies for free in one scan.

The non-saturating calibration remains the one genuinely new property found in this investigation,
and it is the property a ranking-only search is least able to use: at an effective weight of 1000
a monotone rescaling of `h` reorders nothing.

**Why the fold never reached the search, which was left open above and is now settled.** The fold
enters the leaf ranking only through the outside context, and that context charges *the other*
siblings of a leaf -- not all the precursors of its application. For a two-precursor reaction "the
others" is a single molecule, and **sum over a singleton is max over a singleton**. Constructed
minimally, with one application offering two leaves at 0.5 and another offering one at 0.9:

    2 precursors   max context 0.501   sum context 0.501    identical order
    3 precursors   max context 0.501   sum context 1.001    orders diverge

So the folds agree exactly unless an application has three or more precursors. In the 261 steps
extracted from a route archive the distribution is 85 one-precursor, 171 two-precursor and **5
three-precursor** -- 1.9%. One molecule differing in 47,000 is what that predicts.

The methodological lesson is the reusable part. The offline experiment ranked *applications* by a
fold over *all* their precursors, where max(0.5, 0.5) = 0.5 and sum = 1.0 genuinely differ. The
search ranks *leaves* by a context over *the others*, which for a two-precursor reaction is a
singleton. The two measurements were of different objects, and they coincide precisely in the case
chemistry always produces. **When an offline measure moves and the live search does not, check
first that the offline experiment ranks the objects the search ranks.**

This now also lives in :func:`vertsy.aostar.global_leaves`' docstring, beside its note that the
scalar outside context is an additive upper bound under a bottleneck fold, and was re-verified
against that module after the selectivity merge changed its signature.

**The best candidate found here, and it comes from understanding what the search ranks.** The
ranking is ``total = context + inside``; for a leaf ``inside`` is its own estimate and ``context``
accumulates the estimates of the *other* open siblings above it, with the step cost at 0.001
contributing almost nothing. So a leaf's score is essentially **the sum of the estimate over the
open frontier of the solution graph containing it**.

That makes saturation worse than blunt. If every molecule that is not near stock scores about
0.55, then the total is about 0.55 times the frontier size, and the search is ranking by *how many*
open molecules a graph has rather than how far they are. Killing the floor matters more than
improving the correlation.

Scaling the coefficient by the molecule removes it: ``|m| x (1 - similarity)`` grows with what is
left to build rather than saturating. It is a proxy for what GED measures and not that quantity --
see the note at the top of this docstring. It is the best measure in this table:

    1-Tanimoto (incumbent)          rho(open) +0.432
    1-Tversky b=0                             +0.469
    atoms from stock (Tversky)               +0.512
    atoms from stock (Tanimoto)              +0.518

    per true label 0..6
    1-Tanimoto        0.000 0.423 0.497 0.541 0.570 0.589 0.596   5->6 gap 1.2% of value
    atoms             0.000 7.921 11.10 13.52 15.30 16.43 16.97   5->6 gap 3.2% of value

It vanishes at stock, which heavy atom count alone does not -- that sits at 14.98 for the stock
class, which is why size cannot be a heuristic on its own however well it correlates. And it costs
one heavy-atom count on top of the scan already being run.

Two things to settle before believing it. It is **unbounded**, so swapping a [0, 1] estimate for a
[0, 17] one multiplies the effective weight of weighted A* by about seventeen; a live win could be
the greed changing rather than the ranking, and the two should be separated by also testing a
scale-normalised version. And this is a *population* measurement: by the lesson recorded above, it
has to be checked on the object the search ranks -- leaves scored with their context -- before it
is expected to transfer.

**Verified on the object the search ranks, and it reverses the standing recommendation.** Ranking
leaves through :func:`vertsy.aostar.global_leaves` on real one-level states -- so the context is
computed exactly as the search computes it -- over 114 known-good archive steps:

    1-Tanimoto (incumbent)     mean normalised rank 0.423   median 0.408   top quartile 38%
    1-Tversky b=0                                   0.421          0.444              43%
    atoms from stock                               0.233          0.176              66%

**Tversky beta=0 does not transfer.** Its population gain of +0.037 rho is worth 0.002 of mean
rank here, which is nothing. The candidate this table had been calling its one survivor is
worthless on the object the search actually ranks, and should not be wired in on the strength of
the population number alone.

**Unexplained atoms does transfer**, nearly halving the mean rank and nearly doubling the
top-quartile hit rate. On the root of the unsolved target it ranks a known-good precursor at 0.265
where the incumbent and Tversky both give 0.882.

Two caveats on these numbers. They use the n1 stock rather than the campaign stock, because the
full catalogue as a dense float32 matrix is 8.2 GB -- the packed uint64 form the library itself
uses is 255 MB and is the right fix if this is rerun; all four estimates see the identical stock,
so the comparison is sound even though the absolute ranks are not comparable with runs against a
larger catalogue. And the "normalised" variant is not the control it looks like: dividing by a
constant cannot change a ranking, so it is identical by construction. The confound it was meant to
separate -- that an unbounded estimate multiplies weighted A*'s effective weight by about
seventeen -- only appears in the live search, and is still untested.

**Live result for ``atoms from stock``: it transfers, and it buys efficiency rather than reach.**
Both benchmarks at 64x16, incumbent against it:

    hard-22   21/22 both   seen/target 6228 -> 3106   238 s -> 189 s   best reactions 5.14 -> 5.10
    hard-7      7/7 both   seen/target 4572 -> 2585    34 s ->  26 s   best reactions 6.00 -> 6.00

Zero replay failures either way, identical sampled route counts, and route length unchanged. The
same expansion budget interns **half as many distinct molecules** and finishes about 20% sooner.
Contrast the fold switch, which moved one molecule in 47,000: this one reaches the search
unambiguously.

**Why it sees half is mostly size, not better guidance.** Expansions are equal, so the difference
is new molecules interned per expansion: 6.8 against 3.4. Scaling by ``|m|`` penalises large
molecules, the search therefore expands smaller ones, and fan-out tracks size closely -- rho
+0.731 over 120 molecules, with mean distinct precursors running 11.2 below 20 heavy atoms, 21.7
at 20-30 and 30.0 at 30-40. Preferring smaller molecules is defensible in retrosynthesis, since
smaller usually is nearer the stock, but it is a **cheaper** search rather than a better-guided
one, and the two should not be confused. The ranking gain is real and separately measured -- mean
normalised rank 0.233 against 0.423 on known-good steps -- it is simply not what the halving is
made of.

It does **not** solve more targets. The same target fails. So the honest summary is a cheaper
search of the same reach, not a longer reach.

**RETRACTED -- the eight-step chain the next paragraph was built on does not exist.** The chain was
assembled by validating each *molecule* and assuming each *link*. Checked afterwards, four of its
eight steps correspond to no rule in the library. Everything derived from it goes with it: the
30.6 -> 20.9 -> 12.2 -> 8.2 gradient is a measurement along a path the search cannot take, and the
claim that the unsolved target *needs* an eight-step route was never established at all. No
heuristic or selector work should assume that target is deep. The prior question is not about
ranking: it is whether a route to it exists in the graph under the current templates and stock,
and that has not been asked.

The paragraph is kept below, struck through, because it read as a clean diagnosis and would
otherwise be rediscovered -- which is what this log is for.

    ~~Seeing through functional-group chains: diagnosed correctly, fixed wrongly. Along the eight
    step chain the unsolved target needs, the estimate falls 30.6 -> 20.9 -> 12.2 over the two
    steps that remove fragments and then only 12.2 -> 8.2 over the six that interconvert
    functional groups. Each of those six changes one to three atoms out of thirteen, so the
    estimate barely moves and the search gets no reason to prefer the correct edit. This family
    guides convergent assembly and is blind to linear FGI chains, which is the shape of route a
    deep target needs.~~

The negative result that follows is unaffected: it was measured over the library's real one-to-one
rules, not along the invented chain, and it stands.

**What the target actually needs, checked against the paper this time.** Target 13 is compound 7 of
Zumbrunn et al., ChemMedChem 2025, 20, e202401012 -- a galactopyranose-1-carboxamide, 38 nM on
hGal-3. Its route is in the paper's supporting information, and each step of it was applied to the
molecule and checked, rather than assumed as the invented chain above was:

    I-1 + TMSCN/BF3 -> glycosyl cyanide   MISSING  C-glycosylation; the anomeric C-C bond
    I-2 + aryl alkyne, CuI -> triazole    MISSING  CuAAC *from the azide*
    HCl reflux: nitrile -> acid 2         MISSING  MC 4 gives a primary amide only
    2,2-DMP -> 4,6-acetonide              FIRES    PG 12
    Me2SO4/NaH then LiOH -> I-6           FIRES    MC 58a/58c, MC 43a
    T3P: I-6 + Amine 1 -> compound 7      FIRES    Cpl 1
    acetonide / silyl deprotection        FIRES    PG 12, PG 5 (TBS; the SI uses TMS)

The three gaps are consecutive and are the whole front end: everything from acid 2 onward is
covered, and nothing before it is.

**Corrected after running it: one of those three was the blocker, and it was not the one named.**
Adding all three solves the target in 7 reactions -- but so does adding ``MC 71`` alone. Ablating
the whole carbohydrate file changes nothing: same solve, same length. ``MC 47a``, *alkyl nitrile
by cyanide displacement of a chloride*, was already in the library and disconnects the anomeric
C-C bond perfectly well, because an anomeric carbon is ``[CX4]`` like any other. What no rule
could do was get from the acid to the nitrile, and with the acid unreachable ``MC 47a`` was never
offered the substrate it needed.

The earlier probe asked "can any template break the anomeric C-C bond of **acid 2**" and honestly
answered no. The route does not break that bond on the acid; it breaks it on the nitrile, one FGI
away. Validating a bond in the wrong oxidation state is the same error as validating nodes and
assuming edges, one level down, and it produced a confident claim about the wrong template.

``Gly 1a-c`` stay, for fidelity rather than reach: ``MC 47a`` spells this step as an SN2 on an
alkyl chloride, which at an anomeric centre is mechanistically wrong -- it proceeds through an
oxocarbenium, and that is what decides the stereochemistry. A route reading "cyanide displacement"
at C1 is misleading to a chemist even when it reaches the right molecule. **Hartenfeller does carry a Huisgen, but only spelled from the
alkyl halide**, so it offers a 3-chloro sugar nobody stocks where the route wants 3-azido-3-deoxy
galactose, which is a known building block. Reporting that as "triazole covered" would have hidden
a real gap.

So **this target is unreachable, not badly ranked.** No heuristic, selector or fold changes that.
The three candidate families measured above were each asked to find a route that does not exist in
the graph they were searching. That is the honest reason the reach never moved, and it retires the
target as evidence about ranking. C-glycosylation (glycosyl donor + carbon nucleophile) and nitrile
hydrolysis to the acid are the two templates to add if it is to be solvable at all.

An older raw68k run did report this target solved. Its route builds the pyranose by an epoxide
opening from an acyclic aldehyde, which is why a curated library lost a "solve": the count went
down and the chemistry went up.

The obvious repair does not work. Taking ``min(h(m), min over one-step one-to-one retro precursors
of h)`` -- a lookahead restricted to the 107 rules with a single precursor, so there is no AND to
be optimistic about -- scores **0.284** mean normalised rank against plain atoms from stock'
0.233, and 0.412 against 0.265 at the root of the unsolved target. Removing the AND-optimism left
the OR-optimism, which was the problem: with 107 one-to-one rules nearly every molecule has *some*
interconversion that lands nearer the stock, so the minimum pulls the whole field down together
and discrimination goes with it. This is the third time ``min`` over rules has destroyed a ranking.

**Admissibility does not matter here, also measured.** Unscaled the estimate reaches about 30
against a relaxed-phase step of 1.0, so the anytime improvement phase has no guarantee. Dividing by
64 restores it and looks ranking-neutral, but is not: the greedy phase is unaffected while the
relaxed phase, charging 1.0, becomes dominated *by the step* and turns toward breadth-first. On
hard-7 all three solve 7/7 with a mean of 6.00 reactions, while molecules seen run 4571 incumbent,
**2585** unscaled, 3801 scaled, and seconds run 34, **26**, 35. The guarantee bought no better
route and cost the whole efficiency gain, so the default leaves it off.

Run it with::

    .venv/bin/pytest tests/vertsy/perf_heuristic_signal.py -s -m perf


**Named ``atoms from stock`` since 2026-09-29, and ``unexplained atoms`` before that. Both names
over-claim.** The quantity is ``|m| x (1 - max Tanimoto)``: a **size-scaled distance**, not a count
of atoms. Tanimoto compares fingerprint bits, so this is a molecule size times an unshared-bit
fraction -- it carries atom-ish units and counts nothing. The genuine count is ``|m| - |MCS|``,
measured separately below and rejected on cost. Against that, this proxy is rank-faithful and
scale-inflated: Pearson +0.905 and Spearman +0.832 over 60 molecules against 3,000 stock, with
means 9.06 against 5.57 -- about 1.6x high. Entries written before the rename mean the same thing.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Sequence
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np
import pytest

pytest.importorskip("rdkit.Chem")

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tests.datadirs import external_data_root
from vertsy.interop import aizynth_depth
from vertsy.similarity import (
    ECFP4_BITS,
    ECFP4_RADIUS,
    FINGERPRINTS,
    ecfp4_matrix,
    fingerprint_matrix,
)

pytestmark = [pytest.mark.perf, pytest.mark.external_data]

PAROUTES = external_data_root() / "external" / "paroutes" / "processed"


@lru_cache
def labelled(corpus: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Every molecule in ``corpus``'s reference routes, with its distance to that route's leaves.

    A molecule appearing in several routes keeps the *shallowest* label it was ever seen at, which
    is the closest these labels come to "distance to stock". Conflicts are rare — 13 molecules in
    n1, 30 in n5 — so the choice barely matters, but taking the minimum at least errs toward the
    truth rather than away from it.
    """
    path = PAROUTES / f"{corpus}-routes.json.gz"
    if not path.exists():
        pytest.skip(f"missing dataset: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        references = json.load(handle)

    depths: dict[str, int] = {}
    stack = list(references)
    while stack:
        node = stack.pop()
        if node["type"] == "mol":
            smiles = node["smiles"]
            depth = aizynth_depth(node)
            if depth < depths.get(smiles, 1 << 30):
                depths[smiles] = depth
        stack.extend(node.get("children", ()))

    smiles = tuple(depths)
    return smiles, np.fromiter(
        (depths[key] for key in smiles), dtype=np.int64, count=len(smiles)
    )


@lru_cache
def stock_smiles(corpus: str) -> tuple[str, ...]:
    path = PAROUTES / f"{corpus}-stock.smi.gz"
    if not path.exists():
        pytest.skip(f"missing dataset: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return tuple(line.split()[0] for line in handle if line.strip())


def parsed(smiles: Sequence[str]) -> tuple[list[Any], np.ndarray]:
    """Parse, dropping what RDKit rejects and reporting which rows survived."""
    mols: list[Any] = []
    kept: list[int] = []
    for index, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        if mol is not None and mol.GetNumAtoms():
            mols.append(mol)
            kept.append(index)
    return mols, np.array(kept, dtype=np.int64)


def unfolded_morgan(mol: Any) -> set[int]:
    """The molecule's Morgan environment identifiers, unfolded, cached per molecule object.

    Unfolded because the folded 2048-bit union saturates at this stock size and answers nothing;
    cached because the coverage measure asks for the same molecule's identifiers twice and the
    stock is 13.4k molecules.
    """
    cached = getattr(mol, "_vesma_envs", None)
    if cached is None:
        cached = set(
            rdMolDescriptors.GetMorganFingerprint(
                mol, ECFP4_RADIUS
            ).GetNonzeroElements()
        )
        mol._vesma_envs = cached
    return cached


def spearman(estimate: np.ndarray, label: np.ndarray) -> float:
    """Rank correlation, as `corrcoef` of the ranks. No scipy — the project does not have one."""

    def ranks(values: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(values)).astype(np.float64)

    return float(np.corrcoef(ranks(estimate), ranks(label))[0, 1])


def sa_score_of(mol: Any) -> float:
    """RDKit's synthetic-accessibility score, 1 (easy) to 10 (hard), or 1.0 if it refuses.

    Lives in ``rdkit/Contrib/SA_Score`` rather than the package proper, so the path goes on
    ``sys.path`` once. A neutral 1.0 on failure, which multiplies as the identity.
    """
    import os
    import sys as _sys

    from rdkit.Chem import RDConfig

    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib not in _sys.path:
        _sys.path.append(contrib)
    import sascorer  # type: ignore[import-not-found]

    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return 1.0


def qed_of(mol: Any) -> float:
    """Quantitative estimate of drug-likeness, or 0.0 for anything QED refuses to score."""
    from rdkit.Chem import QED

    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def candidates(mols: list[Any], stock: list[Any]) -> dict[str, np.ndarray]:
    """Every estimate worth comparing, from one shared pair of fingerprint matrices.

    All the fingerprint coefficients are three sums recombined — the intersection, and each side's
    bit count — so they cost one matmul between them rather than one each.
    """
    heavy = np.array([mol.GetNumHeavyAtoms() for mol in mols], dtype=np.float64)
    qed = np.array([qed_of(mol) for mol in mols], dtype=np.float64)
    sa = np.array([sa_score_of(mol) for mol in mols], dtype=np.float64)
    query = ecfp4_matrix(mols)
    reference = ecfp4_matrix(stock)
    query_bits = query.sum(axis=1)
    reference_bits = reference.sum(axis=1)
    shared = query @ reference.T
    union = query_bits[:, None] + reference_bits[None, :] - shared

    tanimoto = shared / np.maximum(union, 1.0)
    distance = 1.0 - tanimoto.max(axis=1)
    dice = 2 * shared / np.maximum(query_bits[:, None] + reference_bits[None, :], 1.0)
    cosine = shared / np.maximum(
        np.sqrt(query_bits[:, None] * reference_bits[None, :]), 1.0
    )
    # Tversky with beta=0 charges only the *query's* unmatched features, which is the question a
    # search is actually asking: how much of this molecule is unaccounted for by anything buyable.
    tversky = shared / np.maximum(shared + (query_bits[:, None] - shared), 1.0)

    # The Tversky family is one knob, so sweep it rather than testing one end. beta=0 charges
    # only the query's unmatched features; beta=1 charges only the reference's, which asks the
    # opposite question -- is this stock molecule wholly present in the query. All are free from
    # the matrix above, so the sweep costs nothing and settles the shape of the family.
    def tversky_alpha(a: float) -> np.ndarray:
        b = 1.0 - a
        denom = (
            shared
            + a * (query_bits[:, None] - shared)
            + b * (reference_bits[None, :] - shared)
        )
        return shared / np.maximum(denom, 1.0)

    # A different question entirely: among stock molecules *wholly contained* in the query,
    # how much of the query does the largest one account for? A molecule eight steps out still
    # contains its building blocks, so unlike similarity this need not vanish with distance.
    # `shared == reference_bits` is bitwise containment of the reference in the query.
    contained = np.where(shared >= reference_bits[None, :], reference_bits[None, :], 0)
    biggest_block = 1.0 - contained.max(axis=1) / np.maximum(query_bits, 1.0)

    # Coverage against the union of stock environments rather than the nearest single molecule.
    # Folded, the union saturates every bit at this stock size and the measure is constant — kept
    # in the table so the finding stops someone reaching for it again.
    covered = reference.sum(axis=0) > 0
    folded = 1.0 - (query @ covered.astype(np.float32)) / np.maximum(query_bits, 1.0)
    environments: set[int] = set()
    for mol in stock:
        environments.update(unfolded_morgan(mol))
    unfolded = np.array(
        [
            len(unfolded_morgan(mol) - environments) / max(len(unfolded_morgan(mol)), 1)
            for mol in mols
        ]
    )

    return {
        "1-Tanimoto (incumbent)": 1.0 - tanimoto.max(axis=1),
        "1-Dice": 1.0 - dice.max(axis=1),
        "1-Cosine": 1.0 - cosine.max(axis=1),
        "1-Tversky b=0": 1.0 - tversky.max(axis=1),
        "1-Tversky a=0.75": 1.0 - tversky_alpha(0.75).max(axis=1),
        "1-Tversky a=0.25": 1.0 - tversky_alpha(0.25).max(axis=1),
        "1-Tversky a=0 (b=1)": 1.0 - tversky_alpha(0.0).max(axis=1),
        "largest contained block": biggest_block,
        # The stock distance scaled by molecule size -- a *proxy* for the atoms the nearest stock
        # molecule cannot supply, not that count: Tanimoto is over fingerprint bits, so this is a
        # size times a fraction. Measured against the real thing, `|m| - |MCS|`: Pearson +0.905,
        # Spearman +0.832, means 9.06 against 5.57, so rank-faithful and about 1.6x high.
        # What it buys is that it cannot saturate -- the search sums h over a solution graph's
        # open frontier, and a coefficient piling up near 0.55 for everything far away makes the
        # ranking count open leaves rather than weigh them.
        "atoms from stock (Tversky)": heavy * (1.0 - tversky.max(axis=1)),
        "atoms from stock (Tanimoto)": heavy * (1.0 - tanimoto.max(axis=1)),
        # A third multiplier for *precursor quality*: is what is left a reasonable molecule, or an
        # awkward one? QED stands in for that. Two directions, because which way it should push is
        # not obvious -- a low-QED precursor might be harder to obtain (inflate the estimate) or
        # simply small and reagent-like (leave it alone).
        "atoms from stock x (1-QED)": heavy
        * (1.0 - tanimoto.max(axis=1))
        * (1.0 - qed),
        "atoms from stock / QED": heavy
        * (1.0 - tanimoto.max(axis=1))
        / np.maximum(qed, 0.05),
        "QED alone": qed,
        # Synthetic accessibility, which unlike QED at least points the right way: 1 is easy to
        # make, 10 is hard, so a hard precursor should cost more. `SA - 1` zeroes the multiplier
        # at the easiest molecules rather than merely shrinking it, which is a different ranking.
        "atoms from stock x SA": heavy * (1.0 - tanimoto.max(axis=1)) * sa,
        "atoms from stock x (SA-1)": heavy * (1.0 - tanimoto.max(axis=1)) * (sa - 1.0),
        # SA spans 1 to 10 while the distance spans 0 to 1, so a bare product is dominated by SA
        # and stops being a distance at all. These couple it gently instead: at most a doubling.
        "atoms from stock x (1+(SA-1)/8)": heavy
        * (1.0 - tanimoto.max(axis=1))
        * (1.0 + (sa - 1.0) / 8.0),
        "atoms from stock x (1+(SA-1)/24)": heavy
        * (1.0 - tanimoto.max(axis=1))
        * (1.0 + (sa - 1.0) / 24.0),
        # Weighted geometric mean, which is the principled way to combine two [0, 1] costs: put
        # both on one scale first, then `(d**x * s**y) ** (1/(x+y))` with the exponents saying how
        # much each is trusted. `sa / 10` is the cost direction (hard is expensive) and never
        # reaches zero; `(sa - 1) / 9` is the natural normalisation but *does*, and a geometric
        # mean with a zero factor is zero -- an easy molecule would score 0 however far from stock
        # it sits. Both are here so the difference shows rather than being argued.
        "geo d:SA 1:1 (sa/10)": heavy * np.sqrt(distance * (sa / 10.0)),
        "geo d:SA 3:1 (sa/10)": heavy * (distance**3 * (sa / 10.0)) ** 0.25,
        "geo d:SA 7:1 (sa/10)": heavy * (distance**7 * (sa / 10.0)) ** 0.125,
        "geo d:SA 15:1 (sa/10)": heavy * (distance**15 * (sa / 10.0)) ** (1 / 16),
        "geo d:SA 31:1 (sa/10)": heavy * (distance**31 * (sa / 10.0)) ** (1 / 32),
        "geo d:SA 3:1 ((sa-1)/9)": heavy
        * (distance**3 * np.maximum((sa - 1.0) / 9.0, 0.0)) ** 0.25,
        # As proposed: a similarity-direction factor, so an *easy* molecule is scaled up. Included
        # to show the direction rather than to argue about it.
        "atoms x ((10-SA)/10)**0.5": heavy * distance * np.sqrt((10.0 - sa) / 10.0),
        # and the cost-direction twin of the same shape
        "atoms x (SA/10)**0.5": heavy * distance * np.sqrt(sa / 10.0),
        "SA alone": sa,
        "uncovered, folded union": folded,
        "uncovered, unfolded": unfolded,
        "heavy atom count": np.array(
            [mol.GetNumHeavyAtoms() for mol in mols], dtype=np.float64
        ),
        "exact mass": np.array([rdMolDescriptors.CalcExactMolWt(mol) for mol in mols]),
    }


def nearest_stock_distance(mols: list[Any], stock: list[Any], kind: str) -> np.ndarray:
    """``1 - max Tanimoto`` to the stock under one fingerprint: the heuristic, per view."""
    query = fingerprint_matrix(mols, kind)
    reference = fingerprint_matrix(stock, kind)
    shared = query @ reference.T
    # Squared norms, not sums: identical on 0/1 input and the only correct form on counts,
    # where `sum(a) < a.a` understates the union and lets Tanimoto exceed one. The same
    # expression `vertsy.similarity.stock_similarity` uses, which is the point -- a probe
    # that scores differently from the search is measuring something else.
    union = (
        (query * query).sum(axis=1)[:, None]
        + (reference * reference).sum(axis=1)[None, :]
        - shared
    )
    return 1.0 - (shared / np.maximum(union, 1.0)).max(axis=1)


def test_perf_fingerprint_agreement() -> None:
    """Do several fingerprints disagree enough for their maximum to be worth computing?

    `vertsy.heuristics.strongest` combines admissible estimates by veto: a molecule scores low only
    if EVERY view calls it close. That is worth a second featurisation exactly to the extent the
    views disagree, so this measures the disagreement directly rather than assuming it — and on the
    open-leaf population, since selection never ranks a stock molecule and including them injects a
    "small implies close" correlation nobody discovered.

    Three numbers decide it. Pairwise rank correlation between views says whether they are the same
    opinion twice. Each view's own correlation with true depth says whether a weaker view would just
    add noise to the max. And the max's correlation says whether the combination beats its parts,
    which is the only question that matters for the search.

    **Measured on n1, and the answer is no.** The views are genuinely decorrelated -- pairwise rho
    from +0.41 to +0.67 on open leaves, nowhere near the ~0.9 that would make them one opinion twice
    -- but every combination scores at or *below* ECFP4 alone against true depth: +0.429 for ECFP4,
    then 0.428 (with MACCS), 0.419 (atompair), 0.413 (path), 0.409 (all four). Monotonically worse
    in rank order of how weak the added view is, which is the mechanism: `max` propagates the most
    pessimistic view, and pessimism from a poor predictor (MACCS at +0.265) is noise overriding
    signal. The classical "max of admissible heuristics dominates" does not rescue it, because at a
    weight of 1 against a step of 0.001 none of these is admissible -- this is weighted A*, where
    domination guarantees nothing.

    The search agrees. Sixteen seeded depth-5 targets at 32x8, step 0.001, max fold: all three arms
    solve **16/16 with identical 3.25-reaction routes**, at 35 s, 63 s and 151 s per target. Only
    longest-linear-sequence moves, 3.38 -> 3.25 for all four views at 4.3x the cost -- which width
    alone buys more cheaply.

    **The spread does not rescue it either**, though it took a second experiment to say so. Used as
    an uncertainty rather than a bound -- ranking on `context + inside + c * spread` -- it is not a
    size artefact (spread correlates +0.041 with heavy atoms, *less* than the estimate's own +0.152)
    and it does carry signal (+0.222 against true depth). But it correlates **+0.613 with the very
    estimate it would re-weight**, so `c > 0` is mostly "raise the heuristic weight" and `c < 0` is
    "lower it" -- an axis already swept and found inert.

    Paired against baseline over 64 seeded depth-5 targets at 4x8, step 0.001, max fold:

        c = 0.0  57/64
        c = +1.0 54/64  (+3 -6)   a wash: binomial p ~ 0.5, reorders without net gain
        c = +2.0 39/64  (+3 -21)  over-greedy
        c = -1.0 35/64  (+0 -22)  anti-greedy, and predictable -- spread rises with distance from
                                  stock, so preferring surprise prefers molecules far from it

    Two targets are rescued reproducibly by both positive arms, so the ~40% of spread that is not
    the estimate does something; it is simply far too little to pay for four featurisations.

    Both halves of the multi-fingerprint idea therefore fail for one reason: the views differ in
    *scale* much more than in *ordering*.
    """
    corpus = os.environ.get("SIGNAL_CORPUS") or "n1"
    limit = int(os.environ.get("SIGNAL_MOLECULES") or 4000)
    kinds = [
        k for k in os.environ.get("SIGNAL_FINGERPRINTS", "").split(",") if k
    ] or sorted(FINGERPRINTS)

    smiles, labels = labelled(corpus)
    if limit:
        keep = np.random.default_rng(0).choice(len(smiles), limit, replace=False)
        smiles = tuple(smiles[int(i)] for i in keep)
        labels = labels[keep]

    started = perf_counter()
    mols, kept = parsed(smiles)
    labels = labels[kept]
    stock, _ = parsed(stock_smiles(corpus))
    views = {kind: nearest_stock_distance(mols, stock, kind) for kind in kinds}
    elapsed = perf_counter() - started

    non_stock = labels > 0
    print(
        f"\n{corpus}: {len(mols)} labelled molecules against {len(stock)} stock, "
        f"{len(kinds)} fingerprints, {elapsed:.1f} s"
    )
    print(f"open leaves (label > 0): {int(non_stock.sum())}")

    print(
        f"\n{'view':>10} {'rho(open)':>10} {'mean':>7} {'sd':>7}  mean estimate per true label 1..6"
    )
    for kind, distance in views.items():
        inner = distance[non_stock]
        row = " ".join(
            (
                f"{distance[labels == value].mean():.3f}"
                if (labels == value).any()
                else "  -  "
            )
            for value in range(1, 7)
        )
        print(
            f"{kind:>10} {spearman(inner, labels[non_stock]):+10.3f} "
            f"{inner.mean():7.3f} {inner.std():7.3f}  {row}"
        )

    print("\npairwise rank correlation between views, open leaves only:")
    print(f"{'':>10} " + " ".join(f"{k:>9}" for k in kinds))
    for a in kinds:
        cells = " ".join(
            f"{spearman(views[a][non_stock], views[b][non_stock]):+9.3f}" for b in kinds
        )
        print(f"{a:>10} {cells}")

    # The combination the search would actually run. Reported against the best single view, because
    # "better than the worst" is not a reason to pay for a second matmul.
    print(f"\n{'combination':>28} {'rho(open)':>10} {'mean':>7} {'sd':>7}")
    best_single = max(
        kinds, key=lambda k: spearman(views[k][non_stock], labels[non_stock])
    )
    for label, chosen in [
        (f"best single ({best_single})", [best_single]),
        ("ecfp4 + maccs", ["ecfp4", "maccs"]),
        ("ecfp4 + path", ["ecfp4", "path"]),
        ("ecfp4 + atompair", ["ecfp4", "atompair"]),
        ("all " + str(len(kinds)), kinds),
    ]:
        if not all(k in views for k in chosen):
            continue
        combined = np.maximum.reduce([views[k] for k in chosen])
        inner = combined[non_stock]
        print(
            f"{label:>28} {spearman(inner, labels[non_stock]):+10.3f} "
            f"{inner.mean():7.3f} {inner.std():7.3f}"
        )

    print(
        "\nRead the pairwise block first. Views correlating above ~0.9 are one opinion twice and"
        "\ntheir maximum is one of them at double the cost. The combination earns its matmul only if"
        "\nits rho beats the best single view -- a higher mean is not a win, it is just a larger"
        "\nnumber, and `strongest` raises the mean by construction."
    )


def test_perf_heuristic_signal() -> None:
    """Print how well each cheap estimate predicts reactions-to-stock."""
    corpus = os.environ.get("SIGNAL_CORPUS") or "n1"
    limit = int(os.environ.get("SIGNAL_MOLECULES") or 0)
    bins = int(os.environ.get("SIGNAL_BINS") or 10)

    smiles, labels = labelled(corpus)
    if limit:
        keep = np.random.default_rng(0).choice(len(smiles), limit, replace=False)
        smiles = tuple(smiles[int(i)] for i in keep)
        labels = labels[keep]

    started = perf_counter()
    mols, kept = parsed(smiles)
    labels = labels[kept]
    stock, _ = parsed(stock_smiles(corpus))
    estimates = candidates(mols, stock)
    elapsed = perf_counter() - started

    print(
        f"\n{corpus}: {len(mols)} labelled molecules against {len(stock)} stock, "
        f"{ECFP4_BITS} bits, {elapsed:.1f} s"
    )
    print(
        f"label distribution "
        f"{ {int(v): int((labels == v).sum()) for v in sorted(set(labels.tolist()))} }"
    )
    print(
        "\nLabels are the depth of ONE recorded route each, so they are upper bounds on distance"
        "\nto stock. Fair for ranking candidates, unfit for fixing an absolute scale."
        f"\nDeciles exclude the {int((labels == 0).sum())} stock molecules, whose estimate is 0 by"
        "\nconstruction and which would otherwise collapse the low bins."
    )

    # `rho(open)` is the number that matters and `rho(all)` is the one that flatters. A stock
    # molecule scores exactly 0 by construction and is small, so including the stock injects a
    # "small implies close" correlation nobody discovered — on n1 that alone lifts Jaccard against
    # the label from +0.43 to +0.78. Selection never ranks a stock molecule: they are not open
    # leaves. So the open-leaf population is the only one whose ranking the search depends on.
    non_stock = labels > 0
    print(
        f"\n{'candidate':26} {'rho(all)':>9} {'rho(open)':>10}  "
        f"mean label by {bins}-quantile of estimate, open leaves only"
    )
    for name, estimate in estimates.items():
        inner = estimate[non_stock]
        edges = np.quantile(inner, np.linspace(0, 1, bins + 1))
        means = []
        for index in range(bins):
            low, high = edges[index], edges[index + 1]
            selected = (
                (inner >= low) & (inner <= high)
                if index == bins - 1
                else (inner >= low) & (inner < high)
            )
            means.append(
                f"{labels[non_stock][selected].mean():.2f}"
                if selected.any()
                else "   -"
            )
        print(
            f"{name:26} {spearman(estimate, labels):+9.3f} "
            f"{spearman(inner, labels[non_stock]):+10.3f}  {' '.join(means)}"
        )

    print(f"\n{'candidate':26}  mean estimate per true label 0..6")
    for name, estimate in estimates.items():
        row = []
        for value in range(7):
            selected = labels == value
            row.append(
                f"{estimate[selected].mean():.3f}" if selected.any() else "    -"
            )
        print(f"{name:26}  {' '.join(row)}")

    # The one number this harness must reproduce: `similarity.py:70` records Jaccard correlating
    # +0.06 with heavy-atom count on n1. It reproduces on the open-leaf population and not on the
    # whole set, which is itself the point — the whole-set figure is the stock artifact, and finding
    # that the recorded number only matches one of the two is what identified it.
    heavy = estimates["heavy atom count"]
    jaccard = estimates["1-Tanimoto (incumbent)"]
    print(
        f"\ncross-check, Jaccard against heavy atoms — "
        f"open leaves {np.corrcoef(jaccard[non_stock], heavy[non_stock])[0, 1]:+.3f}, "
        f"whole set {np.corrcoef(jaccard, heavy)[0, 1]:+.3f}. "
        "similarity.py:70 records +0.06, so the open-leaf column is the comparable one."
    )
