# Forward-template selectivity as an application cost

A retro template says a disconnection is *possible*. Nothing in the scoring said whether the step it
proposes is *selective*, because the only knob that priced a reaction was `rule_cost(rule)` — a
function of the template id alone, blind to the molecule the rule was applied to.

So a route could come out step-wise sound and ordering-wrong. The case that started this reads
forward as `xylitol → chlorinate one primary OH → acetalise`, where a chemist would acetalise first
to block two of five hydroxyls: selectively chlorinating one primary hydroxyl of a free pentitol is
optimistic, and nothing in the graph could say so.

This note is the entry point to the change that fixes it. Three documents carry the rest:
`docs/note-heuristic-search.md` names the insertion point (the application cost `c(s)` in
`V(m) = u(m) + min_s [c(s) + Σ V(p)]`); `tests/data/templates/README.md` holds every measurement and
the forward rule table's own rules; and the module docstrings in `src/vertsy/{costs,selectivity}.py`
carry the algebra.

## The signal

Run the **forward** templates paired with a disconnection on that disconnection's own precursors,
and count the distinct products. One is clean. Several mean the intended reaction happens in more
than one place.

Counting is by distinct product *keys* through the run's own keyer, not by match sites. Xylitol's C1
and C5 give two assignments and one product — they differ by molecular symmetry, which a chemist
does not count twice. That is why a pentitol reads three and not five.

Two channels, because they are different chemistry and deserve different prices:

- **`sites`** — distinct products from the forward rules *paired with this rule*: regio- and
  chemoselectivity of the intended transform.
- **`competitors`** — distinct products from *other* multi-component rules: a different reaction the
  same partners could undergo. Multi-component only, and that is a physical argument rather than a
  convenience: both partners have to be in the pot to compete.

`competitors` ships **measured but not charged**. It fires on 71% of applications during search and
on 37.8% of literature reactions that demonstrably worked, and a signal that common cannot price
anything. Narrowing it needs the `Competes` plug point — a favourability table, a conditions table,
or an allow-list — not more rules.

## Why it is a cost and not a veto

A multiplicative penalty on the step cost, because multiplicativity is the unique shape under which
*one* number is correct in both places that price a step: the search folds costs (`base * m`) and
the report folds utilities (`step_factor ** m`), and those base scales are different on purpose. In
both, the multiplier reads **"this step is worth `m` steps."**

The default shape is `m(n) = n + 1` for `n ≥ 2` — n products' worth of material, plus one
separation. The offset is not decoration. The workaround for an unselective step is protect, react,
deprotect, which is three steps under *either* aggregation (a protection detour is linear, so it
costs the same whether an application's precursors are summed or maxed). A multiplier therefore has
to **exceed** 3 to prefer the detour, and the bare `m(n) = n` ties at exactly `n = 3`, which is the
xylitol case.

## The count domain

One integer per channel per application, with two out-of-band values that earn their keep:

| value | meaning | costs |
| --- | --- | --- |
| `UNPROBED` (-1) | no measurement exists | nothing |
| `SILENT` (0) | the probe ran and the forward chemistry made nothing | `doubted`'s factor |
| 1 | clean | nothing |
| n ≥ 2 | a mixture | `n + 1` steps |

**`UNPROBED` and `SILENT` are not the same fact, and conflating them cost a route.** A correct
pairing cannot be silent: the forward rule is the retro rule's inverse on that application, so if the
retro rule fired the forward rule must, and the intended product is among its outcomes. Silence
therefore says the step probably does not work — and while it was spelled `0`, indistinguishable
from "nobody looked", a Stille whose forward rule wanted bromide beat a Suzuki honestly measured as a
two-product mixture. It won by not being measurable.

`doubted(multiplier, silent=...)` charges silence a factor rather than refusing the branch, because
*probably* is the honest word: the same observation is produced by a retro rule proposing partners
its reaction will not couple and by a forward rule this repo flattened too narrowly, and no product
count separates them. On `competitors`, silence is the good answer and is never charged.

An unprobed application pays nothing **by design**, which is why a run states its own coverage. A
high-scale run over a library with no forward counterparts is not selective — it is unexamined.

### The shape family, and the one law it obeys

Three shapes read that column, and a run names one per channel:

| n | `flat:1.2` | `graded:0.1` | `graded:0.5:2` | `steps` |
| --- | --- | --- | --- | --- |
| ≤ 1 | 1 | 1 | 1 | 1 |
| 2 | 1.2 | 1.1 | 1.5 | 3 |
| 3 | 1.2 | 1.2 | 2 | 4 |
| 7 | 1.2 | 1.6 | 2 | 8 |
| 12 | 1.2 | 2.1 | 2 | 13 |

**A multiplier is non-decreasing in n for n ≥ 1: more products, or more competing reactions, may
cost the same but may never cost less.** Saturating is allowed — that is what `graded`'s cap is for,
and it is what makes `flat` a legal member rather than an exception: it is this law at its coarsest,
one step up at n = 2 and a plateau thereafter. A shape that charged a seven-way ambiguity less than
a two-way one would make a report prefer the messier step, and a *composition* could introduce that
without either half doing so, which is why the law is asserted over `doubted` and `uncurated`
wrappings too (`test_costs.py`). It holds from n = 1 upwards only: the sentinels below are out of
band, so charging `SILENT` above a clean step is not a violation.

`graded` exists because neither end of the family states the right thing about a **competitor**
count. Seven there means *the forward table lists seven reactions these materials could undergo* —
more alarming than a two-way split, nothing like eight steps of synthesis. On the five recommended
steps a veto row condemned over six targets, the counts were 7, 4, 1, 1, 1: under `flat:1.2` three
of the five paid nothing, because the plateau starts at 2, and the seven-way step paid exactly what
a two-way one pays. Flat's single step up also lands at n = 2, which on those targets is where the
*correct* answer sits — `MC 62`, the mCPBA epoxidation, reads two competitors while the glycidyl
Grignard that displaced it reads one — so the cliff is in the worst possible place for this data.

### The sweep that says gradation is not the lever

`AOSTAR_HOST_PENALTY_SWEEP` prices one finished graph several ways — extraction is a pure function
of the graph, so N shapes are N *selections*, measured at ~20 ms each against seconds of search —
and scores each by how many veto rows fire on the steps its top route *recommends*.

Selections, not extractions, and the distinction was worth 50x. This sentence used to say "N
extractions", which invited the reading that extraction as a whole is a rounding error. It is not:
timed per phase, extraction runs at **~28% of the search it follows**, dominated by replay and the
mixture report. What the sweep repeats is only `select_solved_routes`, and that really is tens of
milliseconds. That is the
only criterion available here which the thing being tuned does not produce: the rows are
hand-written chemistry and neither selectivity channel reads them.

On the six targets a judge had said the price made worse, every graded slope halved the hits, 2 to
1, while raising flat from 1.2 to 3 did not move them at all. **It did not replicate.** Those six
were the targets the rows had been written while watching, so the agreement was fitting. On twenty
fresh PaRoutes targets, eighteen reporting, all shapes returning 54 routes:

| shape | veto hits on the top route | on any reported route |
| --- | --- | --- |
| `flat:1.2` | 12 | 42 |
| `graded:0.1` | 12 | 38 |
| `graded:0.2` | 13 | 40 |
| `graded:0.5:3` | 10 | 34 |
| `flat:3` | **9** | 33 |
| `steps` | 11 | 33 |

Gradation at a small slope is indistinguishable from the threshold it was meant to improve on. The
column is monotone in **magnitude**, not in shape — and magnitude is precisely what two blind route
judges tied on (9–7, then 6–6, two-sided p = 1.00), so this proxy and the human verdict disagree
about the one variable the proxy responds to. It is therefore not evidence for a stronger default,
and the default stayed `flat:1.2`. Settling it needs a blind judge on fresh targets, not a third
proxy.

### Which channel is worth pricing, measured one channel at a time

The sweep can price the two channels separately (`AOSTAR_HOST_PENALTY_SWEEP=steps/graded:0.1` is
sites/competitors), which is the question the symmetric runs above could not ask. Over the same 20
PaRoutes targets, 18 reporting, all shapes returning 54 routes, counting steps a report
*recommends* that a hand-written veto row objects to — and with the nine azine rows already
corrected, so the proxy is not being fooled by its own false positives:

| sites | competitors | condemned top | condemned any |
| --- | --- | --- | --- |
| none | none | 5 | 14 |
| `flat:1.2` | none | 2 | 8 |
| `graded:0.1` | none | 2 | 10 |
| `graded:0.3` | none | **1** | 7 |
| `steps` | none | **1** | **5** |
| none | `graded:0.1` | 5 | 17 |
| `flat:1.2` | `flat:1.2` | 2 | 8 |
| `steps` | `graded:0.1` | 2 | 5 |
| `steps` | `steps` | 3 | 7 |

**`sites` carries signal and the response is monotone in its strength**: 5 unpriced, 2 at
`flat:1.2`, 1 at `graded:0.3` or `steps`, and 14 → 5 across all reported routes. Five strengths in
order is better evidence than any pair of them.

**`competitors` carries none by this proxy.** Priced alone it is no better than pricing nothing on
the top route and worse across routes (5/17 against 5/14), and added on top of a sites price it
degrades every arm monotonically: `steps/none` 1, `steps/graded:0.1` 2, `steps/steps` 3. This is
the third independent line pointing the same way — the two blind judges tied on the competitors
price, and the opening diagnosis of this whole thread was that a competitor count partly measures
how well *covered* a reagent class is, so charging it penalises the better-curated chemistry.

The combination the proxy prefers, `sites=steps` with `competitors=none`, is what the defaults were
before they were weakened to `flat:1.2` on both channels. The earlier objection to `sites=steps`
was that it biases harder than the curation supports; part of that curation has since arrived, and
the number moved accordingly. **The defaults are unchanged** pending a blind judge: this proxy has
contradicted a human verdict before on exactly the magnitude axis, and it cannot say whether a
route it moved is a better route.

### Why `competitors` carries no price, measured against the veto rows

The sweep says the channel does not help; this says what it is doing instead. The veto rows are
independent ground truth here — hand-written chemistry that neither channel reads — so the question
is whether a channel fires more often on an application a row condemns than on any application at
all. Over **401,321 applications** on the same 20 targets, from the cross-tabulation
`AOSTAR_HOST_VETO_OVERLAP` already produces:

| channel | fires ≥2 on any application | on one a veto row condemns | enrichment |
| --- | --- | --- | --- |
| `sites` | 15.1% | 46.9% | **3.1×** |
| `competitors` | 38.9% | 48.1% | **1.2×** |

`competitors` fires on two applications in five and is 1.2× enriched where trouble is known to be.
That is a **near-uniform tax**, and a uniform tax reshuffles routes without being about anything —
which is why no magnitude rescued it, the gentlest shape tested charging 1.1 at n = 2. `sites`, on
the same ground truth, is 3.1× enriched while firing on one application in seven.

**It is not noise everywhere, and that is the part worth keeping.** Per row the two channels are
complementary, and the split follows the kind of fault exactly:

| veto row | the fault | `sites` ≥2 | `competitors` ≥2 |
| --- | --- | --- | --- |
| B 12, B 13 | an electrophile lands where an azine is deactivated | 93%, 100% | 27%, 36% |
| B 25, B 26, B 27 | a species no bottle holds | 18%, 11%, 16% | 81%, 61%, 73% |

A site problem is a product count, so `sites` sees it and `competitors` does not. A non-isolable
precursor is bifunctional — many multi-component rules can attack it — and that is the same
property which makes it unstable, so `competitors` sees it and `sites` does not. A flat price over
the whole channel averages the signal and the noise into one tax; and the family where the signal
lives is already **refused** outright by B 23 to B 29, so pricing it there buys nothing a veto has
not already taken.

Two hypotheses died on the way, both recorded because each cost a measurement. **Coverage
asymmetry** is real and was the original diagnosis — a reductive amination of a diamine scores 2
while the mesylate alkylation that replaced it scores 0, because few forward rules describe
mesylates — but it is not the whole story, since the channel is only mildly enriched rather than
anti-correlated. And **synonym inflation** — the thought that `forward_probe` drops the frontier
product (`del products`, right for `sites` and arguably wrong here) so a library reads its own
several templates for one transformation as competition — is simply rare: of 14 bimolecular chosen
steps probed by hand against all 129 multi-component rules, **1** had the intended product among
its competitors.

So the channel is not mispriced and not miscounted. It is **too broad**: it asks "what else could
these two molecules do", which on a drug-like pair is usually several harmless things.
`competitor_probe`'s `competes` hook is the one plug point a narrowed version would enter through,
and the enrichment figure above is the test it would have to pass.

#### The narrowing, built and measured: a good diagnostic and no basis for a price

The split worth trying is by **what fixes the competition**. A competitor attacking a *different*
atom is answered by protection — protect the other group, react, deprotect — a bounded, buyable,
three-step detour the selector can purchase, and the one `sites` already prices at that threshold.
A competitor attacking the **same** atom cannot be protected against, since protecting the site
stops both reactions; it is answered only by a different reagent or a different disconnection, so
its yield split is a rate ratio nothing in the step repertoire removes. That reasoning is why the
same-atom half was expected to carry the signal, and it predicted the wrong thing.

Two channels rather than one filter, because the different-site half still costs yield — it is
cheaper to fix, not free — and a `Competes` can only drop a candidate, never label it. The halves
partition the candidates, so together they cost one pass. Over **421,583 applications** on 20
targets:

| channel | fires ≥2 anywhere | where a veto row condemns | enrichment |
| --- | --- | --- | --- |
| `competitors` (unsplit) | 43.7% | 52.6% | 1.2× |
| `competitors:different-site` | 4.8% | 10.4% | **2.2×** |
| `competitors:same-site` | 39.0% | 44.5% | 1.1× |
| `sites` | 15.2% | 41.8% | 2.8× |

**The split relocated the conclusion rather than overturning it**, and the per-row view is where
that is visible:

| row | condemned | `sites` | same-site | different-site | the fault |
| --- | --- | --- | --- | --- | --- |
| *(base rate)* | 421,583 | 15% | 39% | 5% | — |
| B 12, B 13 | 2686, 2219 | **91%, 100%** | 30%, 33% | 5%, 6% | an electrophile on a deactivated azine |
| B 25, B 27, B 26, B 28 | 3463, 1975, 1951, 476 | 19, 15, 7, 12% | **72, 66, 56, 53%** | 17, 17, 6, 8% | a species no bottle holds |
| B 24 | 1318 | 21% | 0% | **25%** | a carbonic monoester |

`sites` owns the regiochemical faults outright. Both competitor halves point at the non-isolable
family — the same-atom half at 1.4× to 1.8×, the different-atom half at 3.4× to 5× on three rows
but firing on only one application in twenty — and that family is already **refused** by B 23 to
B 29. Pricing either half would charge twice for one fault, which is what `VetoLayer` warns
against in so many words.

The volumes also say the rule-level predicate is **loose rather than wrong**: it calls 39% of
applications same-atom against the unsplit channel's 44%, so nearly nine competitions in ten are
labelled same-atom. That is what an over-inclusive signature looks like — the known confusion being
that an acyl carbon and an alkyl carbon are both aliphatic carbons that break a bond and make one.
The match-level tier, comparing the competitor's *matched atoms* against the intended reaction's
rather than two templates in the abstract, would re-partition it honestly. On this evidence that
would sharpen a measurement of chemistry already refused, so it is worth building for the
**report** — a reader wants to know whether a competitor takes the same atom — and not for a price.

#### Half of what the channel counted could not form

Every competitor figure above — the 38.9% base rate, the 1.2x enrichment, the near-uniform-tax
conclusion, the per-row fault families — was measured on a channel counting products that **cannot
exist**, and reading the cases rather than the aggregates is what showed it.

`competitor_probe` argues from reagent presence: a multi-component rule can fire only if both
partners are there, and they are, since they *are* the precursors, so a hit means these starting
materials could do this to each other. **That was never enforced.** `cmol_union` builds one
disconnected graph, and a two-component pattern matches both of its components *inside one
molecule* whenever that molecule carries both groups — which for drug-like precursors is most of
the time. A Suzuki on a chloropyrimidine duly offered, as competitors, a four-membered N-bridged
ring, a benzoxirene, and a gem-diol cyclopropane. Measured over 40 bimolecular steps and 98
competitor products: **55% were made inside one molecule.**

`require_span` subtracts what each starting material does **alone**, which is exact rather than a
proxy — a match firing on one molecule by itself never needed the partner, and a product holding
atoms from both can never appear in a single-molecule pass. It needs no notion of components in the
matcher. Groups of one member are left alone, since there the pot *is* the molecule.

**The declarative fix has since landed, and `require_span` is gone.** `(A).(B)` now parses and is
enforced on CPU, CUDA and Metal; `template_arities` counts zero-level groups rather than dots, so a
rule written `(A.B)` is arity 1 and never enters the channel at all; and all 129 multi-component
forward rows were migrated, so a grouped pattern matched against one molecule produces nothing for
a solo pass to subtract. The subtraction is therefore provably empty, and measured empty: over
**15,601 applications on 20 PaRoutes targets**, `AOSTAR_HOST_SPAN=1` and `=0` gave byte-identical
cross-tabulations.

It removes considerably more than the span did, on the same corpus and the same targets:

| | `competitors` fires ≥2 | on a condemned application | enrichment |
| --- | --- | --- | --- |
| as counted before | 24.0% | 55.9% | 2.3× |
| `require_span` | 17.1% | 40.3% | 2.4× |
| grouping | **9.4%** | 32.8% | **3.5×** |

The extra comes from a case the span could not reach and was never meant to: it skipped groups of
one member, on the sound grounds that where the pot is a single molecule it *is* that molecule and
subtracting would leave nothing. So a **unimolecular** application kept whatever the bimolecular
rule found inside its one precursor. A bimolecular rule cannot fire when there is one molecule in
the pot, so the honest count there is zero, and the guard was holding the defect open on exactly
the applications the subtraction could not touch. `silent` rises from 8,949 to 12,841 of 15,601
and `many` falls from 3,538 to 1,461.

**A second batch of sixteen rules, and the specificity it bought.** Same 20 targets, same
effort, same rule sets:

| | applications | `competitors` fires ≥2 | on a condemned application | enrichment |
| --- | --- | --- | --- | --- |
| before any migration | 14,764 | 24.0% | 55.9% | 2.3× |
| `require_span` | 14,764 | 17.1% | 40.3% | 2.4× |
| grouping, first batch | 15,601 | 9.4% | 32.8% | 3.5× |
| grouping, second batch | 14,776 | **8.8%** | 33.6% | **3.8×** |

The split channel sharpens the same way: `competitors:same-site` reaches **4.5×** from 2.4×, while
`different-site` holds at 2.3×. The channel went from firing on one application in four to one in
eleven, and from 2.3× to 3.8× enriched where the hand-written veto rows — which neither channel
reads — say there is trouble.

The application count is the other half of that story. The first batch raised it by 837, which
looked like the nine new sibling rules and was mostly something else: `iSCB 63a/b/c` and `iSCB 68a`
were character-for-character `Hartenfeller 29a/b/c` and `Hartenfeller 47`, the loader had been
deduplicating them first-wins, and grouping only the Hartenfeller side stopped that. Migrating them
identically restores it, and the count returns to 14,776 — twelve above where it started. So the
migration is close to application-neutral and the 837 was a defect, now covered by
`template_lint.half_migrated_duplicates` and a standing test.

`sites` moves too, and the honest reading is that it moves within nothing: 1.0× before, 0.6× after
the first batch, 0.7× after the second, on a channel whose baseline enrichment in this
configuration is 1.0×. Its unprobed share is back below where it started (2,757 against 2,840),
which was the measurable part of the first batch's effect on it.

**Two gaps the migration opened, both named.** The nine intramolecular retro siblings
(`MC 99`–`MC 103`) needed forward partners or their applications would be `UNPROBED` in both
channels; five rows were written, so `FWD MC 100` covers `MC 100a/b/c` the way its bimolecular
parent covers all three halogens. And the 373 retro rules that are *still* ungrouped keep handing
back one precursor while claiming two, and their now-grouped forward partners correctly decline to
fire on it — which reads as `silent` where it used to read as a site count.
`vertsy.template_lint.ungrouped_components` is the worklist, and
`test_no_forward_rule_leaves_its_left_hand_side_unstated` is the guard that one ungrouped
bimolecular row cannot be added without a test failing.

### Two more fixes, and only one of them improves the enrichment

Same 20 targets, same effort, same rule sets; 14,776 applications in all three arms, and the
`sites` columns byte-identical across them, which is the check that the first fix is
competitor-only as intended.

| | `competitors` ≥2 | on a condemned application | enrichment |
| --- | --- | --- | --- |
| grouping, second batch | 8.8% | 33.6% | 3.8× |
| + same-product filter | 8.1% | 27.1% | **3.4×** |
| + conditions | 6.5% | 23.5% | 3.6× |

**The conditions filter is a clear win.** It removes a fifth of the channel's volume (`many`
1,193 → 958) and only an eighth of its condemned volume (150 → 130), so enrichment rises on all
three channels at once: 3.4× → 3.6× overall, 4.3× → 4.5× same-site, and 2.0× → **2.4×** on
different-site, which gains most. That is the implicit reagent going away: `FWD H 45` no longer
competes wherever an alkyl halide meets a carbonyl, only where the intended step is itself
carbanion chemistry.

**The same-product filter costs enrichment, and is still right.** It removes 7.7% of the
channel's volume but 19.4% of its condemned volume, so 3.8× falls to 3.4×. The outcomes it drops
were *enriched* where the hand-written rows say there is trouble, which is worth stating plainly
rather than burying: this fix removed real signal.

It removed it because the signal was not what the channel claims to measure. The rows that condemn
on this corpus are the non-isolable-species family, `B 25` to `B 29`, and a precursor that is
bifunctional or unstable is one many rules can attack — including two rules reaching the same bond
by different mechanisms, which is exactly the Buchwald-against-SNAr case. So "how many rules reach
the intended product" is a genuine signal *about the precursor*, and it was riding inside a count
of alternative products where it does not belong. A product identical to the intended product
cannot reduce the yield of the intended product, whatever it correlates with.

If that signal is wanted it should be its own measurable — distinct *rules* reaching the intended
product, which is a statement about how many ways the pot can go right — rather than inflating a
channel that prices going wrong. Reverting the filter to recover 0.4× would be optimising the
number against its meaning.

From the start of this work: 24.0% → 6.5% firing, 2.3× → 3.6× enriched. The channel fires on one
application in fifteen rather than one in four.

### Declaring by-products, and what it cost the channels

Nothing. **Twenty-one** rows declared their by-products, one more had its right-hand side
grouped, and the cross-tabulation over 20 targets came back **byte-identical** -- same firing
rates, same enrichments, and the same buckets to the row: `many` 958, `one` 791, `silent` 13,027
before and after. The prerequisite is what
makes that true: the same-product filter tests whether an outcome *contains* the intended product
rather than equalling it, so an outcome that made the product beside its water is still not a
competition. Without that the declarations would have inflated the channel by exactly the number
of rows declaring.

So the counting channels are insulated from the reporting work, which is the condition for doing
the rest of it.

Two earlier readings of this were wrong and are recorded because the error was in the method, not
the chemistry. A run that reported the declarations moving enrichment from 3.6x to 12.8x was
measuring a template file **being rewritten underneath it** -- the edit script replaces the whole
CSV, and the bench had read a partial one. And a run described as covering ten declarations covered
six: the table is loaded once at setup, and four rows landed after that arm had already read it.
Both are why the number above comes from an arm run against a committed tree with nothing else in
flight.

**The rule for which rows may declare: where the pairing names the reagent.** 70 retro rules name
theirs outright. Where the *forward* row also names it on its left-hand side, declaring the
by-product closes the balance exactly -- nothing lost, nothing gained -- which is what
`FWD MC 41a/b/c` (succinimide from N-halosuccinimide), `FWD MC 42` (water from nitric acid),
`FWD MC 84a/b` and `FWD MC 97a/b` now do. Eleven rows are exactly balanced where three were.

Where only the *retro* side names it, the forward rule invents the reagent on its right by the
table's own convention, and declaring the by-products makes what it **gains** equal the reagent's
formula exactly: `C4H6O3` for acetic anhydride, `C10H18O5` for Boc anhydride, `C3H6O` for acetone.
That is a complete account of the stoichiometry, and it is checkable -- a test asserts it per row.
It is also the better place to stop: moving the reagent onto the left-hand side *would* make the
rule atom-strict in the narrow sense, and it would also change the rule's arity and admit it to the
competitor channel, which is a decision about chemistry rather than about bookkeeping.

`FGI 1a` is why the rule has a second half. It is "alkyl halide from a primary alcohol" and names
no reagent anywhere, so declaring `SO2` would be choosing thionyl chloride over `PCl3` or oxalyl
chloride -- each with a different by-product. That is not reporting what is known; it is picking a
reagent on the table's behalf. Nine rows stay underivable from the other direction, where the
leaving species is a workup question: `FWD H 33` loses `Br2` and `FWD MC 85a` loses `Al`, and
`ZnBr2` or `Zn(OH)2` is not something a template can settle.

**What is in the channel now, read as molecules.** Over 25 targets: 646 bimolecular steps, 590
competitor products, of which **39 (6.6%) are the step's own product** — two rules describing one
reaction (`MC 75` and `iSCB 70` are the same esterification under two codes, each appearing as the
other's competitor) or two mechanisms reaching one bond (`Hartenfeller 51g` Buchwald against
`FWD H 54r` SNAr, where an azine substrate admits both). A product that *is* the intended product
cannot compete with itself, so this is a filter worth having and a pairing worth fixing; neither is
done here.

The other 551 are different products, and reading them shows the failure mode has changed kind. The
molecules are molecules now — no benzoxirene, no gem-diol cyclopropane — and the real finds are
real: N- versus O-alkylation of a carbamate against a phenol, an aryl halide choosing between an
amine and a hydroxypropyl chain, azole N-arylation against a Suzuki, a phenol esterifying a free
acid instead of the intended amide coupling. What remains wrong is **the implicit reagent**.
`FWD H 45`, "Grignard addition to a carbonyl", takes a bare `[CX4]-[Cl,Br,I]` as its nucleophile,
so any pot holding an alkyl halide and a carbonyl reads as a competition — including one where the
halide is `OCCCCl`, which would quench the organometallic it never became. `FWD H 30ck` does the
same with the hydride, so a guanidine counts as a reductive-amination partner. Twelve rules supply
77% of the channel and the top four are of this kind.

That is the same lever the table already pulled in the other direction: "naming the reagent is what
lets a rule be a competitor at all", and fifteen rows moved from unimolecular to bimolecular when
the retro table started naming reagents. The next worklist item is the reverse — the rules whose
reagent is still implicit, which answer "could this fire in this pot" for a pot missing the third
thing they need.

A caution on the absolute numbers above: they come from a 20-target run at `16x4` over
`hartenfeller,medchem,protecting-groups,selenium,carbohydrate,phosphorus`, not from the
401,321-application configuration the enrichment table further down reports. The three rows are
comparable to each other and not to that table.

**And the matches are not all artefacts**, which is the honest half of this. `NCCCCC(=O)Cl` closes
to a six-membered lactam, and an intramolecular cyclisation needs no second reagent — unlike the
unimolecular rules this channel excludes by design, the very conditions that couple A to B will
cyclise A. So it is a *genuine competing fate*, merely a different one, and what makes the examples
above worthless is that nothing checks ring size. Measuring intramolecular competition properly is
a real option, and what it needs is the geometry this repository does not yet have anywhere.

#### The enrichment re-measured on possible products, which settles it

With the span enforced, over the same 20 targets — `sites` comes out bit-identical, which is the
control working, since the filter touches only the competitor channels:

| channel | before: base / condemned / enrichment | span enforced |
| --- | --- | --- |
| `competitors` | 43.7% / 52.6% / 1.2× | 30.9% / 33.8% / **1.1×** |
| `competitors:same-site` | 39.0% / 44.5% / 1.1× | 27.6% / 27.3% / **1.0×** |
| `competitors:different-site` | 4.8% / 10.4% / 2.2× | 3.0% / 6.1% / **2.0×** |
| `sites` | 15.2% / 41.8% / 2.8× | 15.2% / 41.7% / 2.8× |

A quarter of the volume was impossible chemistry and is gone, and **the enrichment did not rise.**
The same-site half is now a *perfect null*: it fires on 27.6% of applications and 27.3% of
condemned ones. So the earlier conclusion survives its own premise being corrected, which is the
first time it has actually been tested.

**And the per-row view says why no part of the channel earns a price**, which is the point the
aggregates could not make. Enrichment against each channel's own base rate:

| row | the fault | `sites` | same-site | different-site | instrument |
| --- | --- | --- | --- | --- | --- |
| B 12 | azine halogenation, ortho | **6.0×** | 0.2× | 1.0× | **a price, only** |
| B 13 | azine halogenation, meta | **6.6×** | 0.2× | 1.1× | **a price, only** |
| B 2 | a scoped incompatibility | **3.5×** | 0.1× | 0.0× | **a price, only** |
| B 24 | a carbonic monoester | 1.4× | 0.0× | **3.8×** | already refused |
| B 27 | a beta-halo amine | 1.0× | 2.1× | **3.2×** | already refused |
| B 33 | an unstabilised enol | 1.3× | 0.5× | **3.1×** | already refused |
| B 25 | a carbamic acid | 1.3× | 1.7× | **2.9×** | already refused |

Every row the competitor channel is enriched on is a row **B 23 to B 34 already refuse**. On the
three rows where a price is the only instrument available it reads 1.0×, 1.1× and 0.0× — blind.
`sites` is the exact mirror, at 6.0× and 6.6× precisely there.

So the division of labour between a cost and a wall is now a measurement rather than an argument:
**the competitor channel's signal lives behind the wall.** It detects species already refused and
is blind to the faults only a cost can reach, which is why `sites=steps` with `competitors=none`
is where the defaults sit.

**A note on the declarative route, which pays off beyond this channel.** A retro right-hand
side's dot is "no bond", not a precursor count, so a retro rule written as a bimolecular
disconnection can legally produce *one* molecule by opening a ring rather than cutting a bond in
two — the same defect in the other direction, and the class hand-fixed when three macrocycle rules
"deleted the tether they were cutting". Component grouping turns that into a postcondition the
engine enforces rather than a bug found by reading output: a rule written `>>(B).(C)` that hands
back one molecule is refused, and one written `>>(B.C)` states the ring opening it meant.

**What the sweep did find is a coverage gap no price can close.** `B 13` and `B 14` — electrophilic
halogenation of an azine with the nitrogen meta and para to the halogenated carbon, scoped to
`MC 41a/b/c` — account for 8 of the 12 hits under `flat:1.2` and are present under *every* shape
with near-identical counts; on four targets exactly one row fires, identically, whatever the price.
A multiplier can only move a report toward an alternative that exists, so a row that survives every
shape is saying the halogenation rules have no acceptable alternative to offer on those targets.

### Severity: the half of the competitor channel that is not blind (2026-10-23)

The conclusion above — the competitor channel's signal lives behind the wall — was measured on a
channel that counts *distinct products a pot could make* and makes no distinction between **you
will have to separate two products** and **your substrate is gone**. Those are not the same fact.
A separation has a price; a destroyed substrate does not have one at any price. So the channel was
averaging a cost and a catastrophe, and the aggregate could not say so.

`severity` is one bit per forward row — blank for competing, `destructive` for a row that consumes
the substrate — and `competitors:destructive` is the sub-channel it names, derived in
`_competes_for` from the channel's name exactly as the site halves are. The algebra is untouched:
a channel already carries its own curve in the price's `Factors`, which is the whole reason this is
a channel and not a label on a count.

**Two rules had to be repaired before any of this could be measured.** `FWD DEG 1` and `FWD DEG 2`
each declared *three* left-hand components — the boronic acid, NBS, and water — and no pot that
holds an N-halosuccinimide also holds water: the six retro rules that emit NBS (`MC 41b`,
`MC 84a/b`, `MC 97b`, `FGI 5b`, `FGI 6b`) emit none, and every water-emitting row in the tables is
a hydrolysis or a deprotection. Two components against three declared, so **neither row had ever
matched anything**, and the earlier reading that they fired on the boronic-acid cases was wrong.
Both now name the halosuccinimide as their reagent and let the boron leave as metaboric acid or its
halo ester. Four rows were added: N-halogenation of an amine, aniline or amide N–H by an
N-halosuccinimide (the fault the hard-target reports kept producing, and previously describable by
**none** of the 204 rules — no row formed an N–halogen bond), the hypochlorite N-chlorination, and
the SOCl2 and POCl3 dehydrations of a primary amide.

**Measured, and the division of labour inverts.** 320,309 applications, 20 PaRoutes targets at
`64x16` over `hartenfeller,medchem,protecting-groups`, with the veto rows as independent ground
truth. At the `>= 2` threshold every figure above was taken at:

| channel | reading | fires | on one a veto row condemns | enrichment |
| --- | --- | --- | --- | --- |
| `sites` | two or more products | 14.37% | 17.26% | 1.2× |
| `competitors` | two or more products | 2.02% | 5.59% | 2.8× |
| `competitors:destructive` | two or more **destructive** products | 0.04% | 0.23% | 6.0× |
| `competitors:severity` | **any** destruction | 0.46% | 1.30% | **2.8×** |

**The last row is the honest one, and the row above it is a trap worth naming.** Every figure in
this note is taken at "two or more", because on a count channel that is where a competition starts:
`drop_intended` makes `0` clean, so `1` is one competitor and the threshold has to sit above it.
Applied to the destructive *count*, that threshold asks "did two or more destructive products
appear", and 6.0× is the enrichment of **multiplicity of destruction** — a quantity nobody should
price. A substrate consumed once is consumed. Measuring the event instead — 0.46% of applications
against 1.30% of condemned ones — gives **2.8×**, the same enrichment as the whole channel on a base
rate 4.4× smaller. The volumes say it plainly: 1,354 applications had exactly one destructive product and
122 had several, and the severity reading calls those one fact.

So severity does not beat the count channel on aggregate enrichment. What it does is fire four
times less often for the same enrichment, and — the part that matters — it fires on the rows a
price is the only instrument for.

**The per-row view is the result**, because it lands precisely where the aggregate above said a
price was blind. Enrichment against each channel's own base rate, at `>= 1`:

| row | the fault | `competitors` | `competitors:destructive` | instrument |
| --- | --- | --- | --- | --- |
| B 1 | boron does not survive NBS | 10.8× | **169.8×** | **a price, only** |
| B 12 | azine halogenation, ortho | 1.6× | **18.8×** | **a price, only** |
| B 13 | azine halogenation, meta | 0.8× | **8.8×** | **a price, only** |
| B 2 | a scoped incompatibility | 0.0× | 0.0× | a price, only — still blind |
| B 25 | a carbamic acid | 3.0× | 1.5× | already refused |
| B 26 | a non-isolable species | 3.8× | 1.1× | already refused |
| B 27 | a beta-halo amine | 4.0× | 2.2× | already refused |
| B 28 | a non-isolable species | 3.7× | 0.6× | already refused |
| B 29 | a non-isolable species | 3.9× | 0.0× | already refused |

Read it against the table one section up. The unsplit channel is enriched on the `B 24`–`B 29`
family that `B 23` to `B 34` **already refuse**, and reads 1.6× and 0.8× on the azine rows where a
price is the only available instrument. The destructive band is the mirror image: **18.8× and 8.8×
on exactly those two rows**, and 0.6× to 2.2× on the refused family it is nearly silent about. So
it does not charge twice for what a wall already took, and it is not blind where only a cost can
reach. That is the property the whole channel failed, and it is why splitting on severity was worth
doing where splitting on site was not.

Two honest caveats. `B 1` has only 23 condemned applications, so 169.8× is a direction and not a
figure — what it says is that the positive control fires, which matters because `B 1` is the same
chemistry `FWD DEG 1` and `FWD DEG 2` describe. And `B 2` is still 0.0×: a scoped incompatibility
is not a degradation, and no severity band will make it one.

**Every row earns its place, and none is a uniform tax.** Per forward row over 6 targets:

| row | offered | fired | hit rate |
| --- | --- | --- | --- |
| `FWD DEG 3` N-halogenation by NXS | 4,918 | **479** | 9.7% |
| `FWD DEG 4` N-chlorination by hypochlorite | 861 | 336 | 39.0% |
| `FWD DEG 1` protodeboronation | 4,918 | 38 | 0.8% |
| `FWD DEG 2` ipso-halodeboronation | 4,918 | 38 | 0.8% |
| `FWD DEG 6` amide dehydration by POCl3 | 4,918 | 9 | 0.2% |
| `FWD DEG 5` amide dehydration by SOCl2 | 4,918 | 7 | 0.1% |

`FWD DEG 3` carries the channel, which is the right answer: the N–halogen gap was the fault the
reports surfaced. `FWD DEG 4` is offered only in the two hypochlorite pots and fires on 39% of
them — the subset whose substrate carries a free N–H, not all of them, which is the test the
**bystander invariant** sets. A destructive row's substrate group has to be a casualty and not the
intended reaction's own partner: a Suzuki protodeboronation row would fire on ~100% of Suzuki
applications, because there the boronic acid *is* the coupling's partner, and that is a uniform tax
on one rule family rather than a signal. No row here shows that signature.

**Nothing is priced, and the routes are unchanged.** `YIELD_COMPETITORS` stays `none` and the band
carries no curve. The control: `AOSTAR_HOST_SEVERITY=0` and `=1` over the same targets return
identical `unique_routes`, identical `best_reactions` and identical solve — so every number above
is a measurement of the same search, not of a different one. On this evidence the band is worth
pricing, and what that needs is in the next paragraph rather than in this one.

**The column that makes it priceable: a severity, not a count.** Multiplicity is not the quantity
here. A pot can make several weak side products and still deliver the one you wanted; one reaction
that consumes the substrate delivers nothing. `severity_outcomes` is the reduction that says so --
the same hit rows `distinct_outcomes` reads, reduced to the **worst band that fired** rather than
how many products came out -- and its scale starts at `1`:

| value | meaning |
| --- | --- |
| `1` | nothing fired, or only ordinary competition |
| `2` | a destructive competitor fired |
| `3…` | reserved for a later split by degradation efficiency |

**Starting at one is what makes it priceable, and it is arithmetic rather than taste.**
`clean_log_cost` charges `price.charge(rule, (1,) * len(price.channels))` as the price's free
reference, so whatever a channel stores at `1` costs nothing by construction. On a count channel `1`
is *one competitor* — the very event a severity price must charge — because `drop_intended` correctly
makes `0` the clean reading there. Shifting the measurement rather than bending the curve is what
lets `competitors` keep its own zero point and `drop_intended` stay exactly as it is.

And `drop_intended` should stay as it is. The asymmetry it encodes is right in both directions: a
competitor that remakes the intended product is not competing with anything, while on `sites` the
intended product *is* what is being counted — it is the first of the places the intended reagent
attacked, and dropping it would turn a perfectly selective step from one into zero. The severity
reduction applies the same drop for the same reason: a candidate that remade the molecule the step
was for destroyed nothing, so it contributes no severity.

`ranks` is the severity ordinal per forward rule, and a rule outside it ranks `0` — so an unknown
rule cannot invent severity, which is the opposite of the guard that made the site split
unmeasurable.

**Multiplicity still has a home, and it is the mixture report.** What is in the flask is a question
about named molecules with formulas, exact masses and mass-spectrum coincidence tiers, which
`vertsy.mixture` already answers and a count never could. The price asks how bad; the report says
what. Keeping those apart is why the count channels stay measured and unpriced beside the severity
one: the count is the diagnostic that says how often and which rows, and it is what makes the
decision to price severity checkable against the thing it replaces.

**What a curve on it would be.** The prerequisite is met by the column rather than by a flag, so
what remains is the number — and it wants no new one. `typical_yield` on a destructive row is read
as the **degradation efficiency**, so a competitor consuming the substrate with efficiency `e`
leaves `1 - e` for the intended reaction, which is the share a chemist would compute by hand and
`-log` of it is what `compounded` adds.

The severity column also makes that refinement cheaper than the sub-channel-per-band scheme this
note first proposed. Because `ranks` is an **ordinal and not a flag**, a split by efficiency is more
*values* in one column — `2` for the mildest degradation band through `5` for the most complete —
rather than four more channels. One `Factor` over five points, one probe, no growth in
`shape_of`'s distinct-row count. The ordering is what a band vocabulary is for, and an ordinal is
how a column carries an ordering.

**Found while wiring it, and fixed here.** `competitor_probe` hardcoded `channel="competitors"` on
the `ProbeReport` it emitted, so every sub-channel's report claimed the base name:
`rule_attribution` found nothing under `competitors:same-site` and summed all three sub-channels
under `competitors`. The selectivity columns were always keyed by the real channel name, so the
enrichment tables in this note were never affected and only the per-rule attribution was wrong.

### The chemistry first: which nitrogen, and relative to what (2026-10-23)

Two things were wrong with the first priced version, and both were chemistry rather than wiring.

**`FWD DEG 3` was over-broad, not under-broad.** It matched `[NX3;H1,H2]` — any aliphatic N–H — and
three different fates hide under that:

* **a primary or secondary aliphatic amine** gives an N-haloamine, which eliminates to the imine or
  oxidises onward. Irreversible: the substrate is gone.
* **an amide, sulfonamide or lactam** gives an N-haloamide, which is *what NBS, NCS and NIS are*.
  That is a halogen reservoir, not a sink; the N–X is transferable and the equilibrium returns.
* **an aniline or anilide** gives an N-haloanilide, which **rearranges to the ring-halogenated
  product** (Orton). The material comes back; at worst the regiochemistry is not the one you asked
  for, which is a selectivity question and not destruction.

So the tautomer artefact the last section recorded — a lactam read as destroyed, its aromatic
tautomer read as clean — was never a hole to plug. It was a *symptom* of a rule that should not have
matched either form. The fix is to narrow to the nitrogen class where N-halogenation is
irreversible, after which both tautomers read clean for the right reason, with no veto, no change to
the retro rules, and no claim about rate ratios a product count cannot support.

The dialect has no recursive SMARTS to exclude an amide neighbour with, so the narrowing is
*positive* and split across rows the way `templates-vetoes.csv` splits its protic disjunction:
`[NX3;H2;D1]-[CX4]` for a primary aliphatic amine and `[NX3;H1;D2](-[CX4])-[CX4]` for a secondary
dialkylamine. Measured against a nine-class panel, the old pattern fired on seven classes including
amide, sulfonamide and aniline; the two new ones fire on exactly their own class and on none of
aniline, amide, sulfonamide, either lactam tautomer, or indole. The three reagent families each get
both, so `DEG 3`, `DEG 4` and `DEG 7` become `3a/3b`, `4a/4b`, `7a/7b`.

**`typical_yield` on a destructive row is not a reinterpretation, and `1 - e` was the wrong shape.**
The column means one thing on every row: *what fraction of material this reaction turns into its own
product*. Applied to a reaction nobody wants, that is how completely it destroys — so 0.90 for a
quantitative N-chlorination. Nothing is reinterpreted, and the earlier framing of this as a second
sense for the column was wrong. What a row cannot hold is *what survives*, because that is
**relative**: it depends on the competitor, and a row knows only its own reaction. Storing a
relative quantity in an absolute per-row column is the error, not the band.

So the conversion belongs in the price, and `1 - e` is the wrong one. It is a **mass balance** — it
assumes the degradation runs first and takes its share — whereas these are *parallel* competitors
splitting one substrate by their rates. What survives is `y / (y + e)`, with standalone yield as the
only proxy for a rate these tables have. Over the four-band vocabulary both inputs come from, that
spans 0.357 to 0.643 — **two to four and a half typical steps**, against the seven that `1 - 0.80`
charged — and it lets the intended reaction's own strength count, so a robust step resists a weak
degradation better than a hard one does.

Thirteen distinct ratios fall out of a 4×4 grid of class-level priors, which would claim a
resolution nobody measured, so they are grouped into **four bands** for the reason `typical_yield`
has four. `severity_outcomes` computes the band where both reactions are known — the only place a
relative quantity can be computed, since a `Factor` sees one integer and cannot ask what the step
was for — and `vertsy.costs.banded` decodes it. That pairing is the general answer to "a `Factor`
cannot see the rule": compute the comparison at measurement time and let the curve merely read it.

**Measured on the five hard targets, with the two changes separated:**

| | steps consuming the substrate | bands |
| --- | --- | --- |
| broad rules, unpriced | 72 / 580 (12.4%) | all band 2 |
| **narrowed** rules, unpriced | **13 / 580 (2.2%)** | band 2 ×1, band 3 ×12 |
| narrowed rules, **priced** | **1 / 583 (0.2%)** | band 2 ×1 |

**The chemistry did 82% of the work and the price did the rest, and neither cost a step.** Top-route
lengths are 7, 7, 6, 5 in all three runs — where the absolute `1 - e` price had bought detours of
+2 and +1 reactions to achieve less. A gentler, better-shaped charge reorders within the same length
instead of forcing a longer route.

Every surviving flag is a genuine free primary aliphatic amine meeting a halogen source: `MC 97a`
with NCS on `CC1CC(N)CN(...)`, `MC 39` hypochlorite on `NC1CNCC(=O)C1` and `CC(CN)C(=O)C(=O)O`, and
`FGI 1c` with I₂ on `NC(C1CC1)C(F)(F)CI`. No anilines, no amides, no lactams — the narrowing did
exactly what the chemistry said it would. The one step the price leaves standing is that iodination,
at band 2, the mildest: N–I is the weakest of the three bonds and the intended reaction is robust.
A price that left nothing standing would be a wall, and this is a cost.

### Pricing the severity band: it works, and it is honest about why (2026-10-23)

`YIELD_SEVERITY` defaults to `shared:0.2`, and the number is **derived rather than declared**.
`typical_yield` on a `severity=destructive` row is that row's degradation efficiency; the seven such
rows read 0.65, 0.65, 0.80, 0.80, 0.80, 0.90, 0.90, median 0.80. A competitor consuming the
substrate with efficiency `e` leaves `1 - e`, so the share is 0.20 and nobody picked it. That is
what reading the column as an efficiency buys: the argument moves from "is 0.2 the right constant"
to "is this row in the right band", which is a question a chemist can settle.

No floor is needed, and could not help: `shared` is `share ** (n - 1)` and the severity column is a
**band**, never above 2, so the curve is evaluated at exactly one point above clean. A count channel
needs a floor because eight competitors compound; a band has nothing to compound.

It is a large charge — `-log(0.2)` is 1.609 against 0.223 for a typical step, about **seven steps** —
and that is the claim being made: a reagent that consumes your substrate does not cost you a
separation, it costs you the step.

**Measured on the five pistachio hard targets that prompted the whole thread**, at `64x16` over
`hartenfeller,medchem,protecting-groups`:

| | top route steps / how many consume the substrate |
| --- | --- |
| target 1 | 7 / 0 → 7 / 0 |
| target 2 | 7 / **3** → **9 / 0** |
| target 4 | 6 / **1** → **7 / 0** |
| target 5 | 5 / 0 → 5 / 0 |

Across every reported route: **72 of 580 chosen steps consumed the substrate, and 0 of 617 do.**
Solve rate is unchanged at 4 of 5 — target 3 does not solve either way — and the cost is one extra
reaction on target 4 and two on target 2. Zero is not a wall: a step charging seven steps simply
falls outside `ROUTE_COST_SLACK`, so such a route stops being *reported* rather than stopping
existing.

**On target 2 the new chemistry is right, and legibly so.** The unpriced route ran a directed
metalation and iodination, an NCS chlorination, and a second directed metalation — all three on
substrates carrying a free amine or aniline. The priced route installs the halogens first and brings
the amine in afterwards by SNAr (`Hartenfeller 54a`, twice). Introducing the nucleophile after the
halogenation is what a chemist would do, and nothing told the engine to; it followed from the price.

**On target 4 the improvement is partly an artefact, and that matters more than the win.** Both
routes run the same `MC 84a` benzylic bromination. Unpriced, it runs on `O=C1NCCc2c(Br)cc(CBr)cc21`,
whose aliphatic lactam N–H `FWD DEG 3` sees and charges. Priced, the engine reorders and runs it on
`O=c1[nH]ccc2c(Br)cc(CBr)cc12` — **the aromatic tautomer of the same compound** — where the rule
reads a clean 1. Measured directly: `[NX3;H1,H2]` is an *aliphatic* nitrogen query, so it matches an
aniline and an aliphatic lactam N–H and matches **none** of an aromatic lactam N–H, an indole N–H or
a pyrrole N–H.

So part of what the price bought on that target is the engine routing around the rule's blind spot.
This is exactly the failure mode already on record for the competitors price — "a larger multiplier
amplifies curation asymmetry rather than sharpening chemistry" — and it is the reason the per-row
attribution and these tautomer checks belong beside any price, not after it.

**Whether to close that gap is a chemistry judgement and is deliberately not made here.** NBS does
N-brominate an aromatic lactam N–H, but those rings are electron-rich and C-halogenation competes;
for indole and pyrrole, C-halogenation dominates outright. So an aromatic-N member of the
destructive family would be a claim about rate ratios that a product count cannot support, which is
the same reason `competitors` counts possible products and not the ratio they come in. Recorded as
the next curation question rather than answered by widening a SMARTS.

### The site split was measuring an index error, and the halves swap (2026-10-23)

`_competes_for` built `same_site` over the 204 **forward** patterns and called it with `intended`,
which is a **retro** rule id — `conditions_allow` states that asymmetry in its own docstring and
reads it correctly. `same_site`'s out-of-range guard returned `True`, so every retro rule beyond the
forward table was labelled same-site unconditionally: 204 patterns against 459 compiled rules means
**255 rules (55.6%)** fell through it. `same_site(300, 0)` returned `True` where `same_site(100, 0)`
returned `False`. That is what produced the 39%-same-atom against 4.8%-different-atom volumes and
the reading that the predicate was "loose rather than wrong" — a majority of the same-atom half was
the guard, not a signature.

**The fix is an argument, not a guard.** `same_site(patterns, describes)` takes a retro rule id →
the forward rule ids that are its **inverse** — `probes` and not `probes ∪ scope`, because a scope
entry is a *different* transformation the same reagent does and its reaction centre is somewhere
else. `ForwardLibrary.describes` is that mapping, built beside `index`. Where no forward row
describes the intended reaction there is no centre to compare, so **both halves abstain**: a
`different_site` builder of its own rather than `not same_site(...)`, because negating one would
send all 154 undescribed rules into the other half. Nothing is waved through by abstaining — the
unsplit `competitors` channel measures every one of those applications, and a diagnostic with no
evidence should say nothing rather than pick a side.

**Re-measured, 320,353 applications on 20 targets, at the `>= 2` threshold:**

| channel | fires on any application | on one a veto row condemns | enrichment |
| --- | --- | --- | --- |
| `competitors` (unsplit) | 2.02% | 5.59% | 2.8× |
| `competitors:same-site` | 1.71% | 4.68% | **2.7×** |
| `competitors:different-site` | 0.17% | 0.31% | **1.8×** |
| `competitors:severity` | 0.46% | 1.30% | **2.8×** |

**The halves swap.** The record read 1.1× same-site against 2.2× different-site and concluded that
"the *different*-atom half being the enriched one" was what the chemistry had not predicted. With
the index fixed it is 2.7× against 1.8×, so the expectation the function was written on — a
competitor at the same atom cannot be protected against and is the one that hurts — was right all
along, and the measurement that appeared to refute it was the defect. The volumes fell with it:
7.11% of applications are same-atom at `>= 1`, not 39%.

**The conclusion survives, and the split is kept for a reason that is not a price.** The same-atom
half reads 2.7× where the whole channel reads 2.8× and carries most of its volume, so splitting
separates nothing a price could use. What separates is **severity**, and not by a bigger ratio —
`competitors:severity` is also 2.8× — but by firing on 0.46% of applications where the channel fires
on 2.02%, and by being enriched on the rows a price is the only instrument for.

The site halves stay measured anyway, because the bit they carry is about what to **do** rather than
what to charge. A competitor at a different atom is answered by protect, react, deprotect — a
bounded three-step detour — and one at the same atom is not, because protecting the site stops both
reactions. An engine that inserts a protection sequence into a route needs exactly that bit to know
whether inserting one would help at all, and measuring it now is what makes the feature possible
later. That is a better reason to keep the split than any enrichment figure — and it is what makes
the coverage cap below worth returning to, because the half where protection helps is the thinly
measured one: 3,790 firings against the same-atom half's 22,791.

**Where the intended centre comes from caps the coverage, and the obvious way to widen it does not
work.** It is read off the forward rows that describe a rule, which covers 305 of 459. The retro
template has a centre of its own and would cover 452 — tried, measured, reverted. Two things are
wrong with it and only the first is fixable:

* **The bond roles mirror.** A disconnection breaks what its coupling makes. On `MC 84a` against
  `FWD MC 84a`, exact inverses, the benzylic carbon reads `loses_heavy_bond=True, gains_bond=False`
  one way and the opposite the other. Since `_same_atom` compares those flags for equality, a raw
  retro centre matches **nothing** — the exact mirror of the guard that once matched everything.
  Swapping the two flags fixes this much.
* **The frame does not mirror, and no swap fixes it.** A retro rule's left-hand side is the
  *product*, so its centre describes each atom as it is **after** the reaction, while every
  candidate is a forward rule describing atoms as they are **in the pot**. Over the 271 rules with
  exactly one describing forward row, even after mirroring the roles only **86** centres coincide;
  22 differ in the aromaticity mask alone and **163 differ in element or role**. `Hartenfeller 2a`
  is the shape of it — the nitrogen is an aliphatic amine in the pot and an aromatic ring nitrogen
  in the benzimidazole, genuinely a different query on each side.

So the retro route is not a shortcut to wider coverage; it is a frame error that happens to have
wider coverage, and the second measurement is what tells those apart. It is the same fact that makes
`templates-forward.csv` exist at all: **a retro SMARTS cannot be reversed mechanically.** Pinned by
`test_a_retro_templates_centre_is_not_in_the_candidates_frame` so it is not retried.

**The honest way to widen it is the match-level tier**, which this note has named from the start:
compare the competitor's *matched atoms* with the intended reaction's, rather than the templates'
centres. Matched atoms are frame-correct by construction, because both matches are against the same
pot. What it needs is for the forward matcher to return assignments — today `forward.match` hands
back `(group, rule, application, precursor)` and no atom indices — and a way to put the intended
reaction's match in that frame, which for a retro rule means mapping through the product. Real
plumbing, and nothing here has built it.

**It was in the mixture report too, and that was worse.** `step_mixture` ran the same predicate to
bucket a reported step's peaks, so every candidate of an undescribed intended rule was printed as
"another reaction, same site" — a claim about chemistry, in a chemist-facing report, derived from an
out-of-range guard. The site decision there is now **three-way** — same, different, or unknown —
which gives the mixture a **fourth** rule group and `PeakKind` a **seventh** value, `"another
reaction, site unknown"`. A peak is still a peak and is still reported; it is simply no longer
labelled with a site nobody derived.

`_competes_for` had no test at all, which is how this survived, and the one test covering the
predicate directly asserted the faulty behaviour — "a predicate that cannot judge must not silently
clear: the default is the crude reading". That default was not crude, it was one half of a two-way
split. There are now tests on both.

## Where it plugs in

Two measurement axes, because the payload has exactly two id spaces:

```python
type MoleculeScore[M] = Callable[[MoleculeBatch[M]], Column]            # per precursor row
type ReactionScore[M] = Callable[[MoleculeBatch[M], HitArray], Column]  # per hit row

match = measured(
    expansion(matcher.match_keyed),
    molecules={"estimate": heuristic},
    reactions={"sites": site_probe(...), "competitors": competitor_probe(...)},
)
```

`measured` is the only wrapper, names are given at the call site, and an absent name is a
measurement nobody took — which is not the same as one that came back neutral. A backend that needs
no measurements gets none.

The probe never holds a molecule. It is handed the batch and a **CSR grouping**, so a packed device
batch works exactly as an object array does, and only WL hashes cross the host boundary. `Combine`
answers `(batch, buildable)`: which groups a backend could actually build is the one fact only the
combiner has, and a group over a device's atom budget stays `UNPROBED` rather than reading as
silent.

## What is curation and what is scoring

The single most useful thing learned here: **the counterexample that looked like a missing scoring
axis was a broken pairing.**

`FWD H 43` required `[Br,I]` while its retro partner emits an aryl chloride, so the probe measured
zero products, and free silence made the Stille look clean. It read as an argument for a curated
reagent-preference column ranking tin below boron. It was nothing of the kind, and such a column
would have produced the right answer for the wrong reason while leaving thousands of other silent
pairings in place.

Two lints exist because of that, and they see different things:

- `test_a_pairing_that_can_be_composed_does_compose` composes each pairing against the forward row's
  **own example** — cheap, runs in the suite, catches a pairing whose halves disagree about that one
  molecule.
- `make bench-silent-pairings` runs every pairing over **real reference products** — the only way to
  see a substrate-specific mismatch, and the worklist that took silence from 13.8% to 7.6%.

A curated per-template feasibility prior remains unbuilt and is not needed yet. `by_rule` takes one
whenever a case demands it; a pairwise "wins/loses" relation adds nothing a scalar order does not
already give; "competes with" is the existing `Competes` predicate; and a substrate-dependent
feasibility is a *channel*, not a column — which is what silence turned out to be.

## What it does, measured

Two sweeps, each ranked twice against the **same graph** — once priced, once not. Extraction is a
pure function of the graph, so the two reports differ by the price and by nothing else: no second
search, no second matcher pass, no confound.

| | ten shallow targets (2-6 reactions) | thirteen deep targets (6-9) |
| --- | --- | --- |
| solved | 9 / 10 | 11 / 13 |
| top route changed | 3 | 6 |
| of those, the molecules changed | 2 | 6 |
| routes that got longer | 1 | 2 |

At depth the price moves most reports and every changed top is a changed synthesis. On the shallow
set a third of the changes were the price crediting the more selective of two co-canonical
templates — real, and not a reordered synthesis, which is why both flags are reported.

The two cases worth reading are in `tests/data/templates/README.md`: a Suzuki onto a di-halide
azaindole replaced by a mono-halide coupling with the chloride installed last, and the chloro-tetraol
abandoning "chlorinate one hydroxyl of free xylitol" for a route that forms the C–C bond with the
chloride already in place. Both pay a step to make an ambiguous step unambiguous.

## Known limits, stated

- **The protection detour never wins**, even though `PG 9r`-`PG 12r` now let a route end in a
  deprotection. That agrees with the xylitol measurement rather than contradicting it: an acetal on
  a pentitol is not regioselective either. The adders exist so this is a measurement rather than an
  artifact of the template set.
- **A product count cannot hold a ratio.** The literature xylitol route works by thermodynamic
  control and crystallisation — information about *which* product dominates, which counting possible
  products cannot contain by construction.
- **The false-positive rate on `sites` is 17.1%** against reactions that demonstrably worked. Some
  of that is the probe being wrong; some is selectivity the chemist bought with an order of
  addition, a temperature, or one equivalent.
- **Under the anytime schedule the penalty is near-inert while greedy**, where a step costs 0.001 —
  so the search is reordered in the relaxed phase and the report unconditionally. `surcharged`
  anchors the charge to a fixed scale where that is not wanted.
- **Recall is 21.5%.** Four fifths of real reactions are not expressible by 96 rules, and every one
  of those is unprobed and pays nothing: harmless, and invisible.


## Why protection still does not appear (2026-09-29)

Measured on hard-22 target 13, whose published route protects throughout and whose searched route
protects nothing. Three findings, in the order they were established.

**Competing hydroxyls are already the `sites` channel.** No new penalty type is called for: this
module's own motivating example is chlorinating one hydroxyl of a free pentitol, which is the same
shape as methylating one hydroxyl of free galactose. The economics were already right too --
`steps_equivalent(5) = 6` against a three-step protect/react/deprotect detour, so the detour should
win by three.

**Coverage was a real gap, and closing it changed nothing.** Zero of the seventy-odd `MC` rules had
a forward partner, so `MC 58a` -- the unselective methylation the route actually uses -- was
`UNPROBED`, which by design costs base. `FWD MC 58` now pairs it, and the probe counts what it
should: five distinct products on D-galactose, three on its 4,6-acetonide. Coverage went from
13,469 probed applications to 15,426. **The route did not move.**

**The penalty cannot reach the search, and this is structural.** `selective` is scale-free by
design, which its docstring already names as its one limitation. Under `anytime_step` the greedy
rung charges 0.001 a step, so a sixfold penalty is 0.006 against a heuristic in [0, 1]. Raising
the step until the penalty is legible makes the target unreachable -- swept on this target, with
the sites channel priced and everything else held:

    greedy step   solved   reactions   molecules seen
    0.001         yes      7           11,209
    0.01          yes      7            9,725
    0.05          no       --           9,009
    0.1           no       --           8,944
    1.0           no       --           (breadth-first)

There is no window. Any step cost large enough for a selectivity penalty to matter is large enough
to stop the search reaching a seven-step target. The fix is not a bigger multiplier or another
channel; it is that a multiplier on the step cost and an estimate in [0, 1] are not commensurable,
so either the penalty has to enter as an additive term on the same scale as the heuristic, or the
heuristic has to be denominated in steps.

**A caveat on any result read through partial coverage.** With 19,432 of 32,901 applications
unpaired, the sites penalty is partly a gradient toward unprobed rules. Pricing it made this
target's route abandon `FGI 4a` (probed, and a polyol chlorination it is right to dislike) in
favour of `MC 73a` and `MC 74a` (unprobed, therefore free). The chemistry it landed on happens to
match the paper's -- mesylate, azide, CuAAC -- but the mechanism that moved it was the asymmetry,
not a judgement that mesylate is cleaner. An earlier note in this session claimed that switch as
the penalty working; it is at best half that.


## The commensurability shot, and why it missed (2026-09-29)

The idea worth testing was that the step cost should be scaled to the heuristic's dynamic range
rather than chosen absolutely. `stock` is `1 - Tanimoto` and lives in [0, 1]; `unexplained atoms`
multiplies that by heavy-atom count and reaches about 30. If the balance point is
`range / expected depth`, then a seven-step target wants a step near 0.14 under `stock` and near 4
under `unexplained` -- two very different numbers, and only one of them can be the reason the
default is 0.001.

It is neither. Sweeping the greedy step on target 13 with the sites channel priced:

    heuristic      greedy step    solved
    stock          0.001          yes
    stock          0.01           yes
    stock          0.05           no
    stock          0.1            no
    stock          1.0            no
    unexplained    0.001          yes
    unexplained    0.5            no
    unexplained    2.0            no
    unexplained    5.0            no

The larger dynamic range buys nothing, and above the threshold `molecules seen` is identical at
0.5, 2.0 and 5.0 -- the search is not exploring differently at different weights, it has simply
stopped following the heuristic. So the ceiling is not a scale mismatch that a better unit would
fix. **The search solves only in the essentially-greedy regime**, which is weighted A* at an
effective weight near a thousand, and any `g` term large enough to carry a selectivity signal is
large enough to lose the solve.

That makes this a heuristic problem wearing a cost problem's clothes. A penalty can only reorder
what `g` reorders, `g` is only consulted when the heuristic is not overwhelming it, and this
heuristic has to overwhelm it for the target to be reached at all. Pricing selectivity through the
cost channel cannot work until the heuristic is strong enough to search with a real step cost --
which is the same conclusion, from the other side, as the admissibility measurement in
`perf_heuristic_signal.py`: scaling the estimate to restore the guarantee turned the relaxed phase
breadth-first and cost the whole efficiency gain.

The remaining option, not attempted here, is to stop routing selectivity through `g` at all and
let it enter the leaf estimate instead, where it would be commensurable with the heuristic by
construction. That needs a penalty expressed in the heuristic's units rather than in steps, and
the honest version of that is a heuristic that predicts remaining *work* rather than remaining
*similarity*. Worth writing down as the direction; nothing here measures it.
