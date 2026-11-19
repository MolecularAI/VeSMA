# Route replay

Depending on the rules/templates used,
it might be possible or impossible to recover a route.
Replay could be used to study such theoretical possibility:
given a route, can matcher and templates repeat the route exactly?

Replay have another important role:
search operates on molecular keys `K`,
and keys are not necessarily interpretable
or convertable back to molecules `M`.
To get routes' molecules,
we can replay the route using a readable/interpretable keyer.

## Datasets, and success ratios of route replay and route recovery

This repo uses PaRoutes templates as the main benchmark and reference.
Full raw PaRoutes set is `raw68k`.
All reactions that appear in n1 and n5 routes are `used14k`,
they are much fewer than the full raw set.
Subset `uspto42k` was created in PaRoutes to be distinct from the templates used in n1 and n5 routes. Some templates still can produce the same precursors as the exact raw/used templates, but some disconnections are genuinely not possible to replicate exactly using uspto42k rules.

Here is the table summarizing reactions and templates counts per data set:

| Dataset    | Reactions | Unique templates | 
|------------|-----------|------------------|
| raw68k     |      820k |              68k |
| used14k    |       56k |              14k |
| uspto42k   |       n/a |              42k |


We have precursor-based route comparison metrics that takes into account only precursors, ignoring template/rule id. Precursor order and multiplicity is ignored. So if some template applied to a product can produce identical precursors, we count that as agreement. For routes to agree, we need all steps in the route to agree. We also implemented route replay mode that uses precursor-only comparison: it matches the target using all rules, but keeps only the precursor set that matches the reference route precursor set. It is trimming "fanout" to be exactly 1, or 0 if no identical precursor set exists, thus declaring replay unsuccessful.

Measuring theoretically possible route recovery higlights the importance of choosing reaction templates to cover all desired chemistry if using template-based methods.

## Route replay vs search

Replay starts with SMILES of a target and a route with rule ids, 
and outputs a restricted Expansions[K] object,
from which solved routes can be enumerated.
Route equality is not part of replay.
From target SMILES, we can compute M and K that seed the replay.

Replay can looks like search(), but compute selective_match pairs by looking up frontier keys K in the mapped reference route.

It will pull frontier keys K, lookup rule_id in mapped_route[key], build a list of (mol_idx, rule_id) pairs, and feed them to selective_match.
It expands exactly like vertsy.search.step, interning keys and adding layers.
It returns a restricted Expansions object.

## Route Comparison

We pre-map the reference route to the target backend's native K space (using map_route_mols and the appropriate keyer).
We will run replay_search to get the restricted Expansions.
We will check if the mapped reference route is in solved_routes(target, expansions, stock) to verify identity. Or, we can implement the "loose identity" check (same roots, same leaves, same reactions).

Route replays can use different matcher than the reference route, and keyers might be not available. E.g. replaying route on CPU, while reference route was created on GPU, and GPU keyer is not available for CPU path, or the other way around. We might have routes generated using one set of rules (e.g. PaRoutes raw820k), while trying to compare routes found by search (not replay) using different set of rules (e.g. uspto42k).


### Strict Route Comparison

Map reference route to K space of given Expansions. Traverse two routes and check rule identity, and every molecule identity: both target, intermediates and starting material.

This one is useful if we have exact route. It will not work if we have route keyed by a keyer that is non-reversible, and we don't have access to the exactly the same keyer. For those cases, we need another comparators.


### "Loose" Route Comparison

Run target identity check and reactions identity check, and compare starting materials (leafs) being identical in K-space, or in SMILES space: we can have bidirectional relation for keys in the stock, or decodable map for keys for stock. Then if target was produced from the same starting materials using the same reactions, we can say the route is identical, even if intermediates are never checked.

This comparator is useful if routes are represented with a key we don't have in matcher. We can still replay the route, but we can't compare molecular identities. This comparator assumes we can ask external keyer to key all the stock molecules, then we have bidirectional key<-> molecule relationship for all stock molecules, and we can re-key them with the matcher key, then we can compare the leafs. Since we don't know intermediates in advance, we can not ask external keyer to key all the intermediates in advance, only the fixed stock. This comparator will ensure that synthesis route is the same, and route uses the same stock, even if some intermediates might differ.


### "Library" synthesis style route comparison ("Route topology")

We can allow same root/target, same reactions, but different precursors/intermediates and different starting material/leafs. this will say that all molecules in a library are of the same route. they share reactions, but intermediates and starting materials differ. For replay it will say that the route is correct, working, even though precursors are not the same, but we can work it out from available materials anyway.

This comparator is useful if we don't have access to the external keyer at all. From the reference route we see only reactions, but we can't infer identities of intermediates or leafs. This comparator checks that the route reactions are identical, and that's exactly what is usually required for library synthesis, so we can use this comparator for library synthesis compatibility assessment. 


### Starting materials route comparison

We can allow any reactions, as long as starting materials are the same. This is useful if route comes with "foreign" rule ids, but readable precursors or at least readable starting materials (leafs). Or if route has both foreign rule ids and precursor ids, but we have access to foreign keyer and can key the stock, then match those keys to leafs in the reference routes. We can say that two routes are the same if they synthesisze the same target from the same starting materials, even if they take different routes.


### Precursors route comparison

We can allow any reactions, as long as all precursors - both intermediates and starting materials - are the same. This is useful if route comes with "foreign" reaction ids, but readable precursors, including starting materials (leafs). We can say that two routes are the same if they have identical set of precursors on each step. We ignore order and multiplicity of precursors.


### Matrix

| Molecules                    |    Rule ID present | RuleID absent |
|------------------------------|--------------------|---------------|
| Target, intermediates, stock |             Strict |    Precursors |
| Target, stock                |              Loose |    Start. mat |
| Target                       | "Library"/Topology | Solved/Unsolved only | 
