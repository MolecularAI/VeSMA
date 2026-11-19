# Heuristic search over the retrosynthesis graph

Vertsy separates reaction generation from search:

```text
expander:  molecule -> all successful reaction applications and precursor sets
search:    decide which unresolved work to expand next
valuator:  score the graph, applications, partial solutions, and completed routes
```

Fast exhaustive single-step expansion reduces the need for a template-proposal policy, but
it does not remove the combinatorial growth of multi-step search. It enables a useful
alternative: generate many or all successful applications, then rank them using the actual
precursors and their route context.

The detailed algebra and score catalogue live in another document.
This document describes how search algorithms use those values.

## Search object and vocabulary

The expansion graph is a directed AND/OR hypergraph:

- a molecule/product is an **OR node** choosing one reaction application;
- an application is an **AND node** requiring all its precursor molecules; and
- stock membership is the terminal condition.

For an additive route objective:

```text
c(s) = local cost of choosing application s
h(m) = estimate of the remaining cost from open molecule m to stock
V(m) = current exact-or-estimated inside value of molecule m
Q(s) = c(s) + sum V(p) for precursors p of s
```

The application value `Q(s)`, not an isolated precursor score, compares disconnections.
Sibling values are part of the same conjunction. A precursor/context score is useful for
scheduling expansion only after the application has been valued as a whole.

If stock has zero terminal cost, `h` should vanish at stock. For a cheapest-route guarantee,
it must also be admissible (`h` never exceeds the true remaining cost), and preferably
consistent:

```text
h(product) <= c(application) + sum h(precursor)
```

The zero heuristic is the correctness baseline. Weighted or otherwise inadmissible
heuristics can reduce expansion but give up the optimality proof.

Absolute molecular-complexity descriptors can still be useful as application or route costs,
MCTS signals, or secondary tie-breakers. They are not reaction-count heuristics merely
because they correlate with difficulty.

## Scoring insertion points and their contracts

“Heuristic” is often used loosely for every score in a planner. Only some insertion points
take a heuristic in the formal A* sense. The others take costs, bounds, priors, rewards, or
descriptors, and each has a different contract.

| Insertion point | What it controls | Required or desirable properties | Examples |
|---|---|---|---|
| Goal/solvedness test | whether a leaf is closed | exact membership in correctly keyed stock; deterministic | stock/inventory lookup |
| Terminal cost `l(m)` | preference among accepted stock leaves | defined units; compatible with route objective; heuristic at stock must not exceed it | price, lead time, supplier risk |
| Product/OR-node cost `u(m)` | cost of requiring an intermediate | non-negative for standard shortest-path assumptions; occurrence/reuse semantics declared | handling risk, intrinsic intermediate penalty |
| Application/AND-node cost `c(s)` | cost of choosing a reaction and its joint precursor set | local, additive/composable, calibrated; preferably non-negative | unit step, reaction feasibility, precursor burden |
| Open-molecule heuristic `h(m)` | lower bound before the molecule is expanded | same units as objective; goal-consistent; admissible and preferably consistent for optimal search | zero, bounded stock distance, learned cost-to-go |
| Partial-state heuristic `h(S)` | lower bound for all unresolved molecules in A* | preserves a lower bound when requirements combine; handles multiplicity correctly | sum of molecule bounds, learned set value |
| Primary A*/AO* priority | which lower-bound solution is explored next | `g + h` or equivalent bound; comparable across states/applications | tropical inside/outside bound |
| Secondary tie-breaker | order among equal primary bounds | deterministic, cheap; must not override the primary key to preserve its proof | stock similarity, complexity, uncertainty |
| Branch-and-bound pruning | permanently discards work | certified lower bound versus a valid solved upper bound | exact cost bound |
| Beam/greedy ranking | decides which limited candidates survive | correlated with desired outcome; stable enough to tune | application ranker or composite score |
| Soft incomplete-leaf value | partial credit when search stops unsolved | stock is neutral/best; bounded or numerically stable; AND/OR aggregation declared | missing-leaf penalty, learned reachability |
| MCTS/PUCT policy prior | exploration preference among actions | non-negative, normalized, nonzero support where exploration is required | rule frequency, learned application policy |
| MCTS cutoff value | predicted return below an unexpanded state | aligned with backed-up reward; bounded/calibrated; terminal values anchored | learned solve probability or state value |
| MCTS route reward | what simulations optimize | explicit objective and scale; solvedness kept separate | solved bonus, route cost, yield/risk |
| Post-search metric | reporting, filtering, or reranking routes | reproducible definition and aggregation | reaction count, price, diversity, learned route score |

### The formal lower-bound points

A\* and AO\* attach their guarantee only to lower bounds. With zero-cost stock and additive
application cost, a molecule heuristic should satisfy:

```text
h(stock) = 0
h(m) <= true optimal remaining cost V*(m)
h(m) <= c(s) + sum_p h(p)       for every application s of m  # consistency
```

Consistency is stronger than admissibility and prevents values from becoming more optimistic
after one expansion. A state heuristic built by summing molecule lower bounds remains a
lower bound only under the corresponding additive route-tree objective. If the route reuses
one synthesis of a shared intermediate, naïvely summing duplicate unresolved occurrences can
overcharge it.

An absolute complexity descriptor fails the first condition for reaction count and has no
general relationship to the second. A model trained specifically to predict remaining
reactions can use descriptors as inputs, but the resulting cost-to-go predictor—not any raw
descriptor—is the heuristic. Learned heuristics are rarely provably admissible; use them as
tie-breakers or accept weighted/approximate search unless they have been explicitly bounded.

### The objective points

Costs are allowed to change the answer. This is where absolute complexity can be useful:

```text
V(m) =
    l(m)                                                  if m is stock
    u(m) + min_s [c(s) + sum_p V(p)]                     otherwise
```

- `u(m)` is an OR-node charge paid for requiring molecule `m`, independent of which reaction
  is selected.
- `c(s)` is an AND-node charge paid only when application `s` and all its precursors are
  selected.
- `l(m)` is the cost of accepting a particular stock leaf.

A descriptor-derived `u(m)` makes difficult intermediates undesirable. A descriptor-derived
`c(s)` can penalize a difficult joint precursor set or lack of complexity reduction. A
terminal descriptor means even purchasable leaves carry a structural penalty; in practice,
price, availability, lead time, or supplier risk may be more relevant terminal features.

These are route definitions, not search tricks. Once included, the heuristic must lower-bound
the new composite objective and match its terminal convention.

### The policy and evaluation points

Beam search and MCTS may use predictive scores without admissibility:

- a policy prior ranks concrete applications;
- a molecule-selection policy chooses which unresolved requirement to inspect;
- a cutoff value estimates the return of a partially expanded state; and
- a route reward defines the desired terminal outcome.

Here hand-designed or learned scores do not need admissibility, but they still need the right
scope. A policy for `A -> B + C` should see both `B` and `C`; scoring `B` alone is an
expansion-scheduling choice, not an application policy. MCTS values should be bounded and
calibrated to the backed-up reward so their numeric scale does not swamp exploration.

### Current Vertsy hooks

The present code exposes only a subset of these conceptual points:

- `aostar(..., heuristic=...)` accepts a batched open-molecule estimate. Against a unit reaction
  cost and zero-cost stock, this is the formal reaction-count heuristic.
- `valuation(..., rule_cost=...)` can charge a rule, and `aostar` requires it — no default, since a
  step cost and a heuristic's range are one knob seen from two ends and only their ratio decides
  anything. `constant_step(s)` is the shape it takes until per-template data exists; start near
  `weight = 10 * s` and calibrate against `effort.per_reaction`. A score depending on product and
  precursors needs an application-cost callback, not only `rule_cost(rule)`.
- `valuation` currently returns the semiring `one` immediately for stock. A nonzero terminal
  cost requires a terminal-value hook and corresponding changes to solved-route valuation.
- Selection is injected as `Select`, with two implementations: `descent_leaves` marks the best
  solution graph and offers what it exposes, while `global_leaves` ranks every reachable open
  leaf by `delta + rn` and so can fill a batch. The `Rank` argument already chooses which half
  of that value to order on; a separate tie-breaker hook is the least invasive place to try
  structural or learned preferences without changing the primary reaction-count objective.
- MCTS policy, value, and reward hooks do not currently exist; their contracts should be
  designed together rather than inferred from the AO* heuristic callback.

## Catalogue of guidance functions

The useful catalog is organized by function, not by model architecture:

| Function | Input and output | Search consumers | Simple baseline | When learning adds information |
|---|---|---|---|---|
| Expansion/generation policy | product -> candidate templates/applications and priors | any budgeted search | apply every rule; necessary-only structural prefilter | template libraries are too large to apply exhaustively |
| Application filter | product + complete precursor set -> accept or feasibility | any search before insertion | validity, duplicate, cycle, forbidden-motif rules | empirical feasibility, chemoselectivity, forward consistency |
| Application cost or prior | successful application -> cost or normalized preference | A*/AO*, beam, MCTS | unit step, rule frequency, precursor count, stock hits | rank actual outcomes by feasibility, yield, conditions, precedent |
| Molecule cost-to-go | molecule -> remaining cost or solve probability | A*/AO*, greedy search | zero, bounded stock distance, cached graph value | strong guidance under a small expansion budget |
| Partial-state value | unresolved molecule multiset -> expected cost/return | MCTS; state-space A* | sum/max of molecule values | sibling interactions, shared conditions, non-additive risks |
| Molecule scheduling/value of information | target context + open molecule -> expansion priority | batched AO*/best-first control | inside/outside value, occurrence count, uncertainty proxy | predict which expansion will most improve the solution graph |
| Terminal reward | solved or cutoff state -> scalar outcome | MCTS, RL | solved indicator minus declared route costs | empirical success, yield, preference, or downstream utility |
| Route/ensemble scorer | route(s) -> score, rank, or diversity | post-processing, top-k selection | reactions, depth, price, stock sets, structural diversity | experimental route quality or learned human preference |

The first two are easy to confuse. An **expansion policy** controls which actions are even
generated; an **application policy** ranks successful actions after their precursors are
known. Vertsy's fast exhaustive matcher makes the first optional for moderate rule sets and
makes the second more attractive: it can score the actual match site and precursor set
rather than predict them from the product alone.

A learned filter is also different from a learned cost. A hard filter can destroy
completeness; a cost or prior only changes order while alternatives remain reachable. Start
new empirical models as soft scores, measure false negatives, and promote them to pruning
only when that loss is accepted.

## How far simple guidance can go

No learned model is required to:

- generate every application in the rule library;
- enforce graph and molecular validity;
- recognize stock exactly;
- prove a minimum-reaction route when search completes within its declared limits;
- calculate solvability, depth, counts, soft route mass, and inside/outside values; or
- rank routes by explicit properties such as reaction count, price, stock fraction, or
  diversity.

A strong non-learned progression is:

1. unit application cost and zero heuristic;
2. exact stock plus a bounded nearest-stock heuristic;
3. deterministic cycle, duplicate, and chemical-sanity filters;
4. empirical rule/template frequencies and stock metadata;
5. precursor-aware application features such as stock hits, branch count, fragment balance,
   and non-negative complexity progress;
6. inside/outside context and soft route-ensemble usage.

This is enough for an explainable baseline and for generating training/evaluation data.
Learned models become most useful when the missing signal is empirical rather than
structural:

- whether a formally matching reaction is experimentally feasible or selective;
- which successful disconnection is likely to work under realistic conditions;
- how much unresolved synthesis remains beyond a shallow horizon;
- whether sibling branches interact through conditions, protection strategy, or shared
  intermediates; and
- which route qualities correlate with experimental or human preference.

Even there, a table or calibrated regression on reaction precedent may be a better first
step than a large neural model. Compare every model against matched compute budgets and
report expansions saved, solution recall, route quality, calibration, and inference cost.

## Relation to planner interfaces

Existing planners illustrate the same decomposition without prescribing Vertsy's design.
[Syntheseus](https://microsoft.github.io/syntheseus/stable/cli/search/) exposes Retro*,
MCTS, and PDVN as search choices; its examples separate Retro* OR-node cost, AND-node cost,
and value functions, while MCTS receives reward, value, and optional policy functions.
[AiZynthFinder](https://molecularai.github.io/aizynthfinder/configuration.html) defaults to
MCTS with expansion-policy priors and optional filter policies, and also exposes a
[Retro* molecule-cost hook](https://molecularai.github.io/aizynthfinder/aizynthfinder.search.retrostar.html)
plus independent route scorers.

The durable lesson is the interface split, not the current class catalog: reaction
generation, filtering, local cost/prior, cost-to-go/value, reward, and post-search scoring
should remain independently replaceable.

## A* over partial route states

Ordinary A* represents a state as the multiset of unresolved molecules plus the reaction
applications already chosen:

```text
state S = {unresolved molecules}
g(S) = cost of committed applications
h(S) = sum h(m) for unresolved m
f(S) = g(S) + h(S)
```

An action chooses an unresolved molecule and replaces it with the precursors of one
application. The next state is canonicalized so equivalent unresolved multisets share a
queue entry.

A* has a familiar correctness story and works well when:

- complete route-state identity is cheap;
- interaction between unresolved molecules matters;
- the number of distinct frontier multisets remains manageable; and
- a good admissible state heuristic exists.

Its main weakness here is duplication: the same molecular subproblem appears in many
different partial-route states. State multisets can grow combinatorially even when the
underlying molecule graph is highly shared.

Branch-and-bound is a complementary technique: once a solved route supplies an upper bound,
discard any partial state whose lower bound cannot beat it. Alpha-beta pruning is not the
right model; it relies on adversarial min/max turns, whereas retrosynthesis has cooperative
AND requirements under OR choices.

## AO* and Retro*-style search

AO* works directly on the AND/OR graph:

```text
V(stock) = 0
V(open molecule) = h(molecule)
Q(application) = c(application) + sum V(precursor)
V(expanded molecule) = min Q(application)
```

The currently cheapest alternatives define a **best partial solution graph**. AO* expands
an open molecule on that graph, inserts its applications, and propagates changed values
back toward the target. A solved route is an upper bound, not automatically a proof of
optimality: search may stop only when no cheaper open partial solution remains.

Retro*-style planners use this same broad best-first AND/OR idea, often with a learned
molecule value or reaction policy. Neither is required. An exhaustive expander plus a simple
explainable heuristic is a valid configuration.

AO* is the natural default for Vertsy because:

- its values match the stored product/application/precursor relation;
- shared molecular subproblems are valued once;
- the best solution graph includes all siblings of a chosen application;
- those siblings form a useful expansion batch; and
- deferred alternatives remain in the graph for later revaluation.

`src/vertsy/aostar.py` implements this separation with plain functions: prepare and value the
graph under tropical/min-plus, find open leaves on the cheapest solution graph, expand a
bounded batch, and repeat. `src/vertsy/similarity.py` supplies an optional ECFP4/Jaccard
distance-to-stock heuristic; with unit reaction costs and weight at most one, its bounded
range makes it admissible.

Min-plus is important here because `min` selects an application. Softmin/log-sum-exp does
not: it accumulates mass from several alternatives, so there is no unique “best application”
to descend through. Use soft route mass to score the graph or form route posteriors, not as
a drop-in AO* selection algebra.

## MCTS and PUCT

MCTS usually operates on partial route states like A*, but allocates effort through repeated
selection, expansion, evaluation, and backup. PUCT combines an empirical action value with
a policy prior and an exploration bonus:

```text
PUCT(S, a) =
    Q(S, a)
  + exploration_constant * prior(S, a) * sqrt(visits(S)) / (1 + visits(S, a))
```

When all successful applications are already generated, the prior need not propose
templates. It can rank concrete applications:

```text
prior(s | m) proportional to exp(-(c(s) + sum h(p)) / temperature)
```

A cutoff-state value can aggregate unresolved molecules, and a terminal reward can combine
solvedness with declared route-quality terms. SA-score is permissible here because MCTS does
not require admissibility, though its scale and whether it rewards molecules, reactions, or
complete routes must still be explicit.

MCTS is attractive when:

- search should be strongly anytime and exploratory;
- rewards are noisy, learned, or non-additive;
- diverse attempts matter more than proving the cheapest route; or
- uncertainty and stochastic outcomes are part of the objective.

It has more policy, backup, state-hashing, and calibration choices than AO*. Transpositions
also complicate visit accounting: sharing a molecule value is not the same as sharing a
whole partial-route state.

## Metrics and semirings around search

Semiring dynamic programming and heuristic search play different roles:

- **min-plus** supplies a best-route objective and lower bounds for A*/AO*;
- **boolean** reports whether any solved route exists;
- **counting** reports how many routes exist under declared identity and bounds;
- **softmin/log-sum-exp** scores the mass of good alternatives;
- **top-k** retains several good route backpointer structures;
- **Pareto values** retain non-dominated multi-objective alternatives;
- **inside/outside values** rank molecule occurrences with sibling and target context; and
- **expectation or posterior passes** report reaction/intermediate usage across a route
  ensemble.

One search run can build a graph that several valuations inspect afterward. Do not force
every metric into the search priority. A selective scalar objective is easiest to search;
route diversity, soft feasibility, common intermediates, and batch synthesis coherence are
often better computed as post-search graph analytics.

For multiple good routes, continue beyond the first proof or evaluate top-k backpointers.
Repeatedly deleting the current best route is a poor diversity mechanism: it can produce
near-duplicates or arbitrary discontinuities. Generate a candidate set under a clear route
identity, then select for diversity explicitly.

## Cycles and budgets

Retrosynthesis graphs can contain cycles even when extracted routes forbid repeating a
molecule on one path. Search and valuation therefore need declared bounds:

- maximum reaction depth;
- maximum expanded molecules or applications;
- positive/non-negative local costs;
- cycle-safe route extraction;
- saturation for counts; and
- a policy for zero-cost cycles and duplicate applications.

Depth-layering `(molecule, remaining depth)` is a simple way to make valuation well-founded.
Fixpoint or SCC-based methods are alternatives when an unbounded graph value has suitable
convergence properties. A wall-clock cutoff is operationally useful but does not itself
justify an optimality claim.

## Which algorithm to start with

| Need | Good starting point | Main trade-off |
|---|---|---|
| Exhaustive graph and aggregate metrics | breadth expansion + graph DP | builds all work within the horizon |
| Cheapest route with shared subproblems | AO*/Retro*-style min-plus | incremental graph/value bookkeeping |
| Cheapest route with state-level interactions | A* over unresolved multisets | state explosion and duplication |
| Fast approximate route | weighted AO*, weighted A*, or beam search | no cheapest-route guarantee |
| Stochastic exploration or complex reward | MCTS/PUCT | policy/value/backup calibration |
| Several high-quality routes | top-k graph DP plus diversity selection | memory and route identity |

The recommended baseline is:

1. Define a non-negative application cost and terminal semantics.
2. Verify exhaustive or zero-heuristic search on small graphs.
3. Add an admissible, goal-consistent heuristic and measure molecules/applications expanded.
4. Use AO*/Retro*-style graph search when sharing and batching matter.
5. Add soft route mass, top-k, outside values, or MCTS only for a specific downstream need.

The fastest planner is not the one with the most elaborate score. It is the simplest search
whose value meanings, terminal conditions, bounds, and guarantees match the question being
asked.
