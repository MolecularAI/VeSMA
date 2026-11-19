# Route metrics, heuristics, and semirings

This document is the conceptual reference for valuing Vertsy's retrosynthesis expansion
graph. It separates four ideas that are easy to conflate:

- a **metric** measures a completed or partially expanded graph;
- an **objective/cost** states which routes are preferable;
- a **heuristic** estimates the remaining objective before expansion; and
- a **semiring valuation** composes leaf, reaction, and route-alternative values without
  materializing every route.

Search algorithms consume these quantities.

## The expansion graph

A retrosynthetic application is a directed hyperedge:

```text
product molecule -> reaction application -> {precursor 1, ..., precursor k}
```

- A product is an **OR node**: any one application may solve it.
- An application is an **AND node**: all its precursors must be solved.

Vertsy stores the graph as a flat relation in molecule-storage-id (`sid`) space. 
Rows identify a product, rule/application, and precursor;
grouping by application reconstructs AND nodes,
and grouping applications by product reconstructs OR nodes.
Molecules shared by many routes are interned once.

The molecule graph is not necessarily acyclic. Protection/deprotection and other reversible
transformations can form cycles. A concrete route is usually interpreted as a finite tree,
and unfolding a shared cyclic graph into all route trees can be exponential or infinite.
Therefore:

> Compute aggregate values on the shared graph. 
> Materialize only the few routes that a caller actually needs.

## The general valuation

For a semiring-like value domain `(S, ⊕, ⊗, 0̄, 1̄)`, let:

- `⊕` combine alternative applications at an OR node;
- `⊗` combine jointly required precursors and the chosen application at an AND node;
- `leaf(m)` value a terminal or open molecule; and
- `w(s)` value a reaction application.

Then:

```text
V(stock molecule m) = leaf(m)

Q(application s) =
    w(s) ⊗ ⨂ V(p) for p in precursors(s)

V(product m) =
    ⨁ Q(s) for s in applications(m)
```

`V(target)` is “`⊕` over routes of `⊗` over each route.” Swapping the algebra changes the
question while preserving the graph traversal.

A semiring supplies associative operations, identities, and distributivity of `⊗` over
`⊕`. OR aggregation is normally commutative; AND aggregation must also be commutative when
precursor order has no meaning. Some useful domains below are semiring-like dynamic programs
rather than strict scalar semirings, but the same OR/AND decomposition still applies.

In the current implementation, `src/vertsy/metrics.py` provides:

- `TROPICAL`: min-plus cost;
- `BOOLEAN`: solvability;
- `log_semiring(temperature)`: soft route mass; and
- `valuation`: a memoized, depth-bounded graph fold returning both product and application
  values.

The stock value in this fold is the semiring's `one` (`0` for min-plus). Open unexpanded
products receive `leaf_value`; products beyond the depth budget receive `zero`. Separate
specialized functions implement saturated route counts, depth, leaf-set counts, and route
comparators where their representation or cycle semantics do not fit the generic scalar
fold cleanly.

## Catalogue of route valuations

| Question | OR: `⊕` | AND: `⊗` | Terminal value | Important caveat |
|---|---|---|---|---|
| Solvable? | OR | AND | true for stock | least fixpoint on cyclic graphs |
| Cheapest route | min | sum | zero | tropical/min-plus; selective |
| Best route probability | max | product | one | use negative logs to recover min-plus |
| Route count | sum | product | one | cycles require depth bounds or saturation |
| Soft route mass | softmin/log-sum-exp | sum | zero cost | non-selective; depth/fan-out sensitive |
| Minimum reaction count | min | sum plus step cost | zero | counts total reactions in the tree |
| Minimum route depth | min | `1 + max` | zero | bottleneck/deepest branch, not reaction count |
| Minimum missing leaves | min | sum | zero stock, one missing | values approximate/incomplete routes |
| Bottleneck difficulty | min | max | leaf difficulty | optimizes the hardest required branch |
| Top-k routes | k-way merge | k-best Cartesian product | empty route | retain backpointers, deduplicate |
| Pareto routes | non-dominated union | vector combine | zero vector | frontier size needs bounds |
| Expected statistics | expectation semiring | product-like combine | statistic identity | needs normalized route weights |
| Distinct stock sets | set union | pairwise set union | singleton stock set | combinatorial; cap and depth-bound |

### Boolean solvability

```text
solved(stock) = true
solved(application) = all solved(precursor)
solved(product) = any solved(application)
```

This is an exact reachability statement. `solved_sids` evaluates it as a least fixpoint:
seed stock molecules, repeatedly promote applications whose precursors are all solved, and
stop when the solved set no longer grows.

### Tropical/min-plus best route

With non-negative application costs:

```text
V(m) = 0                                             if m is stock
V(m) = min_s [c(s) + sum_p V(p)]                    otherwise
```

This compares whole applications, so siblings are never judged independently:

```text
A -> B + C       costs c1 + V(B) + V(C)
A -> D + E       costs c2 + V(D) + V(E)
```

Min is **selective**: `min(a, b)` returns one of its inputs. That property lets AO* identify
a best application and descend through its open leaves. It is why min-plus can guide search.

Reaction count is the special case `c(s) = 1`. Other costs can represent price, predicted
feasibility, risk, or a deliberate weighted route-quality objective. Units must be coherent:
adding a probability, an SA-score, and a reaction count without calibration does not create
a meaningful cost merely because the program accepts floats.

Application identity matters before scoring. If several templates produce the same canonical
precursor multiset, they may represent one chemical disconnection with several pieces of
template evidence rather than several distinct route branches. Depending on the intended
semantics, combine their costs with:

```text
best supporting template:       c(application) = min_t c(template t)
soft evidence aggregation:      c(application) = -tau * logsumexp_t(-c(t) / tau)
```

The second rewards multiple supporting templates. It is only a calibrated probability if
the evidence model justifies that interpretation; otherwise it is a useful score. Keeping
equivalent templates as separate OR alternatives also inflates route count and soft route
mass, so route identity and template evidence should be designed together.

### Probability and negative-log cost

If `r(s)` is a reaction-success probability and precursor success is treated as independent:

```text
P(application) = r(s) * product_p P(p)
P(product) = max_s P(application)
```

Taking `-log` converts products to sums and max to min, yielding the tropical recurrence.
Replacing max with sum gives total probability mass, but only if alternatives are mutually
exclusive or the probabilistic interpretation accounts for overlap. Otherwise call it a
score or route mass, not a calibrated probability.

### Softmin and log-sum-exp route mass

For route cost `C(r)` and temperature `τ > 0`:

```text
Zτ(m) = sum over routes r from m of exp(-C(r) / τ)
Fτ(m) = -τ log Zτ(m)
```

The recursive cost form is:

```text
Qτ(s) = c(s) + sum_p Fτ(p)
Fτ(m) = -τ log sum_s exp(-Qτ(s) / τ)
```

This is `log_semiring(τ)`. As `τ -> 0`, `Fτ` approaches the best-route min-plus value, and
`log_semiring(0)` is exactly `TROPICAL`.

The critical difference is selectivity:

```text
min(3, 3) = 3
softminτ(3, 3) < 3
```

Two equally good routes carry more mass than one. Softmin therefore answers “how much good
route support exists?”, not “which application is best?” It is useful for smooth
synthesizability scores, route ensembles, and downstream rewards, but it cannot directly
choose the AO* solution graph. Use tropical values to select; use log-sum-exp to score.

Soft route mass rewards multiplicity. Duplicate templates, high fan-out, and cycles can
inflate it even when chemistry has not improved. Canonicalize equivalent applications,
depth-bound or otherwise control cycles, and consider reporting a normalized quantity
alongside total mass.

The implied local route-ensemble posterior is:

```text
Pτ(s | m) = exp((Fτ(m) - Qτ(s)) / τ)
```

It supports expected reaction/intermediate usage, uncertainty, and common-route analysis
without committing to a single best route.

### Counts, top-k, and diversity

The counting semiring uses sum over applications and product over precursor route counts.
A graph cycle permits infinitely many route trees, so practical counts must define their
identity and bound:

- rule-aware routes;
- precursor-set routes, collapsing templates that produce the same precursors;
- starting-material-set routes;
- a maximum depth; and
- a saturation cap.

For several good routes, top-k DP is preferable to repeatedly finding one route and deleting
part of it. At an AND node, lazily combine child k-best lists; at an OR node, merge and retain
the best candidates. Store costs and backpointers rather than full route trees.

“Top-k” is quality, not diversity. Diversity is a separate selection criterion over route
features such as first disconnection, reaction classes, intermediates, or stock sets. A
practical design is to generate a larger cheap candidate pool with top-k DP, then select a
smaller set with a diversity penalty or constraint.

### Product, Pareto, bottleneck, and expectation domains

When two route attributes must stay coupled, carry them together. For example, a shallow
route can have more missing leaves than a deep route; computing the two minima separately
would splice attributes from different routes.

Options include:

- **scalarization:** combine attributes with declared units/weights, then use min-plus;
- **lexicographic value:** optimize the primary metric and use the rest as tie-breakers;
- **Pareto frontier:** retain all non-dominated vectors;
- **bottleneck value:** combine required branches with `max`, useful for depth or hardest
  precursor; and
- **expectation semiring:** carry `(route mass, weighted statistic)` to obtain expected step
  count, template usage, or other additive statistics under a route ensemble.

Pareto and top-k carriers grow with the graph. Pruning, deduplication, caps, and clear route
identity are part of their semantics, not merely implementation details.

## Inside, outside, and sibling-aware priority

An **inside** value describes the subproblem below a molecule:

```text
I(m) = cost or mass from m down to stock
```

An **outside** value describes the target-specific context above one occurrence of it. In
the tropical case, for `s: m -> {p1, ..., pk}`:

```text
O(target) = 0

candidate O(pi) =
    O(m) + c(s) + sum_{j != i} I(pj)
```

The full cost of a route constrained to use that occurrence is `O(pi) + I(pi)`. This is the
right way to ask whether `B` is promising in `A -> B + C`: its score contains sibling `C`,
the chosen application, and alternatives through `A`.

Inside values are reusable across targets; outside values are target- and objective-specific.
With log-sum-exp, the analogous outside pass yields occurrence, reaction, and intermediate
marginals over the soft route ensemble. These can support:

- expansion scheduling by target relevance;
- bottleneck and value-of-information estimates;
- common-intermediate or common-reaction analysis across targets; and
- batch objectives for shared synthesis.

Do not discard one precursor from a multi-precursor application because it scores poorly in
isolation. Applications are route choices; precursor occurrence scores are scheduling hints.

## Metrics, objectives, heuristics, and priorities

These values have different contracts:

| Quantity | Meaning | May change the chosen route? | Terminal requirement |
|---|---|---:|---|
| Metric | Report about a graph/route ensemble | no, unless reused as an objective | defined by the metric |
| Application cost `c(s)` | Cost of choosing one reaction | yes | none |
| Product/node cost `u(m)` | Cost of requiring a molecule occurrence | yes | defined explicitly |
| Terminal cost `l(m)` | Cost of accepting a stock leaf | yes | evaluated at stock |
| Heuristic `h(m)` | Estimate of remaining objective at an open leaf | only through search order if admissible search completes | must match the objective's terminal convention |
| Priority/tie-breaker | Chooses what to expand next | not if used only after the admissible primary key | stock nodes are not expanded |

For the current reaction-count objective:

```text
c(s) = 1
l(stock) = 0
```

A valid molecule heuristic should:

1. **Be goal-consistent:** `h(stock) = 0`.
2. **Be admissible** when optimality is required: `h(m) <= V*(m)`.
3. Preferably be **consistent** for every application:
   `h(m) <= c(s) + sum_p h(p)`.
4. Be non-negative, deterministic, cheap enough to batch, and expressed in the same units
   as the objective.

The zero heuristic is the correctness baseline. An inadmissible or weighted heuristic can be
useful, but then the result is heuristic best-first search rather than a proved cheapest
route.

More generally, a heuristic does not intrinsically have to vanish. It must agree with the
terminal objective:

```text
h(stock m) <= l(m)
```

It vanishes only when the objective assigns zero cost to stock, as the current min-plus fold
does.

## Similarity to stock

`src/vertsy/similarity.py` implements:

```text
h(m) = weight * (1 - max_{b in stock} Tanimoto(ECFP4(m), ECFP4(b)))
```

The implementation uses radius-2, 2048-bit Morgan/ECFP fingerprints 
and exact Jaccard distance against the entire stock matrix
(Jaccard distance $J$ is the complement of Tanimoto similarity $T$: $J = 1 - T$).
Bit intersections are a matrix multiplication;
union sizes follow from the two bit counts and the intersection.

This heuristic has the right terminal behavior: an exact stock molecule has distance zero.
Its range is `[0, weight]`. If every non-stock molecule requires at least one reaction and
the minimum application cost is one, `weight <= 1` is automatically admissible. Larger
weights deliberately trade the cheapest-route guarantee for greedier search.

Similarity is a proxy, not a synthetic distance. The implementation deliberately uses
Jaccard rather than Hamming: Hamming distance strongly reintroduces molecular size because
large fingerprints differ in more positions. A sufficiently wide fingerprint reduces
collisions that can turn even Jaccard into another size proxy.

Exact stock membership and nearest-stock similarity have different authority:

- exact membership decides that a molecule is solved;
- similarity supplies a heuristic, partial-credit score, or diagnostic;
- similarity must never silently promote an analogue to solved stock; and
- stock, target, and generated precursors must use the same molecular key space before exact
  membership is tested.

The last point is a correctness condition, not a scoring detail. If stock retains a different
stereochemical or canonical form from generated precursor keys, buyable molecules become
invisible and the search repeatedly expands them.

## Sources of scoring functions

The valuation does not care whether a leaf or application value came from a rule, a lookup
table, or a neural model. Their semantic contract matters more than their implementation.
There are three useful tiers:

1. **Exact structural logic:** stock membership, graph reachability, cycle checks, atom and
   bond validity, route depth, and duplicate detection. These define correctness and should
   not be replaced by prediction.
2. **Empirical but simple scores:** template frequency, observed success counts, stock price,
   supplier availability, reaction-class preferences, and calibrated descriptor formulas.
   They are data-driven without requiring a learned model.
3. **Learned predictors:** reaction feasibility or selectivity, application ranking,
   molecule cost-to-go, state solve probability, yield/risk, and route quality.

| Quantity | Simple baseline | Possible learned function | Contract that survives either implementation |
|---|---|---|---|
| Application cost | unit cost; precursor count; cycle or motif penalty; template-frequency table | reaction feasibility, forward-likelihood, yield, selectivity, condition compatibility | score the product and complete precursor set; calibrate direction and units; keep non-negative when shortest-path assumptions require it |
| Molecule heuristic | zero; bounded distance to stock; cached exact lower bound | remaining reaction count/cost, probability of reaching stock | terminal-consistent; admissible/consistent if cheapest-route proof is required |
| Application policy | softmax of hand-designed application costs | rank successful applications or templates | non-negative normalized preference; distinguish proposal from pruning |
| State value | sum/max of molecule values; current inside DP | solve probability or expected cost for the whole unresolved set | respect sibling/state context and match the backed-up objective |
| Soft incomplete-leaf value | missing-leaf penalty plus stock distance | calibrated partial-synthesis or reachability score | exact stock remains authoritative; stable range and declared AND aggregation |
| Terminal cost | zero; price, lead time, supplier risk | procurement or availability predictor | evaluated only after the exact terminal test; same units as route cost |
| Route scorer | reactions, depth, stock fraction, price, diversity | yield, experimental success, route preference | preferably post-search unless it deliberately defines the search objective |

Learned models are most valuable where the graph and simple descriptors do not contain the
needed information: empirical reaction feasibility, chemoselectivity, condition
compatibility, and strong cost-to-go estimates under a tight expansion budget. They are
least appropriate for exact stock membership, chemical validity, cycle accounting, or the
algebra that combines alternatives and requirements.

Learned outputs also need ordinary engineering discipline:

- validate out of distribution across targets, rule sets, and stock catalogs;
- separate ranking quality from probability calibration;
- measure inference latency and batching cost against expansions saved;
- avoid train/test leakage through templates, routes, or stock;
- retain a zero/simple baseline and ablate each model;
- never let a low learned score become an irreversible prune unless lost routes are an
  accepted trade-off; and
- record whether optimality/completeness claims were abandoned.

### Absolute molecular-complexity descriptors

SA-score, SCScore, Bertz complexity, heavy-atom count, and ring count are examples of
hand-designed or learned molecular descriptors, not universal search heuristics. They
describe absolute structure and generally remain nonzero for stock molecules, so raw values
do not lower-bound remaining reaction count.

They can still be useful as:

- a secondary tie-breaker after an admissible primary value;
- an input to an application cost, such as a non-negative failure-to-simplify penalty;
- a product or terminal cost when intermediate/stock complexity is deliberately part of the
  objective;
- a feature in a learned cost-to-go, policy, MCTS value, or soft incomplete-route score; or
- a post-search route descriptor.

`max` over precursor complexity asks about the hardest branch; `sum` asks about total burden
and also penalizes precursor count. Charging an absolute value at every open molecule can
therefore bias against multi-precursor or convergent routes. Pick the intended meaning and
aggregation before choosing a descriptor.

Product-and-precursor application features
requires an application-cost callback or precomputed application weights. 
Likewise, nonzero stock costs require a terminal-value hook 
because the generic valuation returns the semiring `one` for stock.

## Partial-credit synthesizability

When no complete route is found, a separate soft metric can give graded credit,
as in SynthSense AiZynthFinder-Reinvent scores.
For example, with missing-leaf penalty `gamma_missing`,
step discount `gamma_step`,
stock-distance penalty `beta`,
and optional application score:

```text
log_leaf(m) =
    0                                                   if m is stock
    log(gamma_missing) - beta * distance_to_stock(m)    otherwise

log_route(route) =
    number_of_steps * log(gamma_step)
  + sum_missing_leaves log_leaf(leaf)
  + sum_steps log_step_score(step)
```

Then aggregate alternative routes with max for the single best approximate route or
log-sum-exp for soft route mass. This yields a smooth score related to expressions such as
`0.2^missing_leaves * 0.8^steps`, while retaining sibling structure and avoiding route
materialization.

This is a metric or downstream reward, not proof of solvability. Depth, missing-leaf count,
similarity, and route multiplicity can all bias it; expose those components alongside a
single scalar during calibration.

## Route recovery and identity

Reference-route recovery is not merely bottom-up scoring. It threads the current reference
position top-down and returns a boolean match bottom-up. The finite reference tree bounds the
recursion even when the expansion graph contains cycles.

Vertsy supports a family of comparators:

| Comparator | Reactions | Intermediates | Stock leaves | Meaning |
|---|:---:|:---:|:---:|---|
| Strict             | yes | yes | yes | same reaction-labelled tree |
| Loose              | yes |  no | yes | same reaction skeleton and starting materials |
| Library            | yes |  no |  no | same reaction skeleton |
| Precursors         |  no | yes | yes | same precursor set, any reactions |
| Starting materials |  no |  no | yes | same stock-leaf set, any reactions |
| Solved             |  no |  no |  no | some route reaches stock |

These comparisons answer different questions when rule ids or molecule identities cannot be
aligned. Route counts must likewise declare whether template-distinct but chemically
identical applications count separately.

## Cycles, bounds, and convergence

There are three common evaluation strategies:

1. **Acyclic/topological fold:** exact and single-pass when a depth-layered or otherwise
   acyclic graph is available.
2. **Depth-bounded memoized fold:** value `(molecule, remaining depth)`; recursion always
   decreases the budget. This is what the generic `valuation` uses.
3. **Fixpoint iteration:** appropriate for monotone, convergent domains such as boolean
   solvability; SCC-aware variants can improve propagation.

Counting and soft route mass can diverge on cycles because repeatedly traversing a cycle
creates more route trees. Positive costs prevent a cycle from improving a tropical minimum
but do not necessarily make the total mass finite. Define maximum depth, simple-route
semantics, caps, or convergence assumptions explicitly for every metric.

Cycle-cut route extraction and graph valuation also answer subtly different questions.
Whether a shared subproblem is reusable under “no molecule repeats on this path” depends on
the ancestry set, so unrestricted memoization is not exact for simple-route enumeration.

## Multiple targets and route ensembles

A shared expansion graph lets targets reuse target-independent inside values. Each target
then needs only its root and an outside/usage overlay.

Under a soft route ensemble, propagate expected usage from each root through application
posteriors. This supports:

- common intermediates or reactions;
- common stock sets;
- late-stage diversification opportunities;
- route-family clustering; and
- rewards for coherent synthesis across a generated batch.

Do not confuse two different objectives:

- **single-target diversity** rewards independent alternatives;
- **multi-target coherence** rewards shared chemistry.

One tends to penalize repeated route features; the other rewards the same features across
different targets.

## Design checklist

Before adding a score, answer:

1. Is it a reported metric, a route objective, a search heuristic, or only a tie-breaker?
2. Does it live on a product, an application, a terminal leaf, or a whole route?
3. How do sibling precursors combine: sum, max, product, or a vector?
4. How do alternative applications combine: min/max, sum, softmin, top-k, or Pareto union?
5. What are `zero`, `one`, the dead-end value, and the stock value?
6. What are the units, range, and direction?
7. Must the heuristic be admissible or consistent?
8. What do cycles, duplicate applications, and route identity mean for it?
9. Is the computation depth-bounded, capped, normalized, or convergent?
10. Can it be computed on the shared graph without materializing routes?

Keeping those choices explicit is more important than finding a single universal
“synthesizability score.” Different questions require different algebras.
