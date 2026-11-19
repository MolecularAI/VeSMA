# Atomic properties from SMARTS

Ullmann algorithm performs graph node labels check.
We will use atom labels based on SMARTS pattern and SMILES string.
Each graph node is an atom, and as a node label we will use all atomic properties that are easy to extract from SMARTS pattern and SMILES string.
Rings require extra ring detection, but we can live with that.
Chirality requires extra processing (e.g. rdchiral).
Bond orders require multiple connectivity graphs in the arc consistency algorithm.
We ignore those features for now.
Instead, we use simple atomic number and charge, degree, total connectivity, total valence and H-counts (total and implicit).

We want a mask for what is allowed according to the SMARTS pattern. Then we match the atom properties of the mol to the pattern mask using bitwise-AND. If the result per property is non-zero, then the property matches. Then we take the logical-AND between all properties to determine if the mol properties matches the pattern.

Let's use one byte for each property, except for atomic number Z, which will use 16 bytes (128 bits).

Let's use the following:
- Z: atomic number, boolean flag in bits 0..127, for atomic numbers 0..127. Atomic number 0 is reserved for "any", but we don't use it.
- A: aliphatic aromatic or any. Boolean flag in bits 0 (aliphatic) and 1 (aromatic), other bits in the byte unused.
- C (+, -, ++, --, -2, -1, -0, +0, +1, +2, etc in SMARTS): Formal Charge: Specifies the net formal charge of an atom, bits for the following charges: [-3 (and more negative), -2, -1, 0, +1, +2, +3, +4 (and more positive)]. Note: practical molecules typically have charges -2..+2, but to use all 8 bits, we use -3..+4.
- D: Explicit Degree: Specifies the number of explicit connections (neighbors), bits for the following degrees: [0, 1, 2, 3, 4, 5, 6, 7 and more].
- H: total hydrogen count, both implicit and explicit, bits for the following hydrogen counts: [0, 1, 2, 3, 4, 5, 6, 7 and more].
- h: Implicit Hydrogen Count: Matches the number of implicit hydrogens attached to an atom. [Ch2] matches a carbon with 2 implicit hydrogens.
- V (v in SMARTS): total valence, bits for the following valences: [0, 1, 2, 3, 4, 5, 6, 7 and more].
- X: Total Connectivity: Specifies the total number of connections (explicitly bonded neighbors), excluding implicit hydrogens, bits for the following connections: [0, 1, 2, 3, 4, 5, 6, 7 and more].
- R: ring membership: if 0 then in no rings, if 1 then in 1 ring and so on until 7, where 7 means in 7 or more rings.
- r: ring size: bit 0 is in no rings (even though "r" without number in SMARTS means "in some ring", so bits 1-7 must be true), bits 1-6 mean in a ring, where smallest ring size is correspondingly 1-6. bit 7 means the smallest ring is of size 7 or more.
- x: ring connectivity: total ring connections (bits correspond to 0..7+).

Let's not use the following from SMARTS:
- @ for chirality (no chirality)
- no recursive SMARTS $(SMARTS)
- ignore atomic mass

## SMARTS Grammar

Reformatted from https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html and https://ics.uci.edu/~dock/manuals/DaylightTheoryManual/theory.smarts.html


### Atomic primitives

- * wildcard:    any atom
- a aromatic:    aromatic
- A aliphatic:   aliphatic
- D<n> degree (default <n>: one):  <n> explicit connections
- H<n> total-H-count (default <n>: one): <n> attached hydrogens
- h<n> implicit-H-count (default <n>: one or more): <n> implicit hydrogens
- R<n> ring membership (if <n> is missing, then in some ring): in <n> SSSR rings
- r<n> ring size (if <n> is missing, then in some ring):   in smallest SSSR ring of size <n>
- v<n> valence (default <n>: one): total bond order <n>
- X<n> connectivity (default <n>: one)    <n> total connections
- x<n> ring connectivity (default <n>: one or more): <n> total ring connections
- -<n>   negative charge: -<n> charge -1 charge (-- is -2, etc)
- +<n>    positive charge: +<n> formal charge  +1 charge (++ is +2, etc)
- #n  atomic number:   atomic number <n>
- @   chirality   anticlockwise
- @@  chirality   clockwise
- @<c><n> chirality:   chiral class <c> chirality <n>
- @<c><n>? chiral or unspec:    chirality <c><n> or unspecified
- <n> atomic mass: explicit atomic mass

Defaults:
- `D`, `H`, `v`, `X` with no number mean exactly `1`.
- `h`, `R`, `r`, `x` with no number mean at least `1`.


### Bonds
- -	single bond (aliphatic)
- /	directional bond "up"1
- \	directional bond "down"
- /?	directional bond "up or unspecified"
- \?	directional bond "down or unspecified"
- =	double bond
- #	triple bond
- :	aromatic bond
- ~	any bond (wildcard)
- @	any ring bond


### Logical Operators

Atom and bond primitive specifications may be combined to form expressions by using logical operators. In the following table, e is an atom or bond SMARTS expression (which may be a primitive). The logical operators are listed in order of decreasing precedence (high precedence operators are evaluated first).

1. "!", aka exclamation, example "!e1" : not e1
2. "&", aka ampersand,   example "e1&e2" :  a1 and e2 (high precedence)
3. ",", aka comma, example "e1,e2" :   e1 or e2
4. ";", aka semicolon, example  "e1;e2" :    a1 and e2 (low precedence)

All atomic expressions which are not simple primitives must be enclosed in brackets. The default operation is & (high precedence "and"), i.e., two adjacent primitives without an intervening logical operator must both be true for the expression (or subexpression) to be true.

The ability to form expressions gives the SMARTS user a great deal of power to specify exactly what is desired. The two forms of the AND operator are used in SMARTS instead of grouping operators.

Examples:
[CH2]   aliphatic carbon with two hydrogens (methylene carbon)
[!C;R]  ( NOT aliphatic carbon ) AND in ring
[!C;!R0]    same as above ("!R0" means not in zero rings)
[n;H1]  H-pyrrole nitrogen
[n&H1]  same as above
[nH1]   same as above
[c,n&H1]    any arom carbon OR H-pyrrole nitrogen
[X3&H0] atom with 3 total bonds and no H's
[c,n;H1]    (arom carbon OR arom nitrogen) and exactly one H
[Cl]    any chlorine atom
[35*]   any atom of mass 35
[35Cl]  chlorine atom of mass 35
[F,Cl,Br,I] the 1st four halogens.


### Component-level grouping of SMARTS

SMARTS may contain "zero-level" parentheses which can be used to group dot-disconnected fragments. This grouping operator allows SMARTS to express more powerful component queries. In general, a single set of parentheses may surround any legal SMARTS expression. Two or more of these expressions may be combined into more complex SMARTS:

- (SMARTS)
- (SMARTS).(SMARTS)
- (SMARTS).SMARTS

The semantics of the "zero-level" parentheses are that all of the atom and bond expressions within a set of zero-level parentheses must match within a single component of the target.

| SMARTS | SMILES | Match behavior |
| --- | --- | --- |
| `C.C` | `CCCC` | yes, no component level grouping specified |
| `(C.C)` | `CCCC` | yes, both carbons in the query match the same component |
| `(C).(C)` | `CCCC` | no, the query must match carbons in two different components |
| `(C).(C)` | `CCCC.CCCC` | yes, the query does match carbons in two different components |
| `(C).C` | `CCCC` | yes, both carbons in the query match the same component |
| `(C).(C).C` | `CCCC.CCCC` | yes, the first two carbons match different components, the third matches a carbon anywhere |

These component-level grouping operators were added specifically for reaction processing. Without this construct, it is impossible to distinguish inter- versus intramolecular reaction queries. For example:

| Reaction SMARTS expression | Match behavior |
| --- | --- |
| `C(=O)O.OCC>>C(=O)OCC.O` | Matches esterifications |
| `(C(=O)O).(OCC)>>C(=O)OCC.O` | Matches intermolecular esterifications |
| `(C(=O)O.OCC)>>C(=O)OCC.O` | Matches intramolecular esterifications (lactonizations) |

#### What vesma implements, which is the table above

Both halves, because the table states both: atoms sharing a non-zero group must land in one
component, and atoms in different groups in different ones. All six rows pass, including the
mixed `(C).C`, and so do the three reaction rows — `(C(=O)O).(OCC)` matches an acid and an alcohol
but not a hydroxy acid, and `(C(=O)O.OCC)` the reverse. Daylight's own rationale is this
project's: the operators exist "specifically for reaction processing", because "without this
construct, it is impossible to distinguish inter- versus intramolecular reaction queries", which
is exactly the ambiguity measured here in both directions.

The three forms:

| written     | means                              | arity |
| ----------- | ---------------------------------- | ----- |
| `A.B`       | unspecified; the dot constrains nothing | ambiguous |
| `(A).(B)`   | two molecules                      | 2     |
| `(A.B)`     | one molecule, two fragments of it  | 1     |

RDKit implements one half and lists the other as unsupported, which is worth being precise about
because the two halves are enforced by different machinery there.

**The binding half is RDKit's documented mechanism.** The RDKit Book's "Intramolecular reactions"
section says they "can be expressed flexibly by including reactants in parentheses" and gives a
ring-closing metathesis, `([C:1]=[C;H2].[C:2]=[C;H2])>>[*:1]=[*:2]`. `ReactionFromSmarts` turns
`([C:1].[N:2])>>` into one reactant template and `([C:1]).([N:2])>>` into two, and vesma's
`template_arities` agrees with it on all 655 shipped templates, both sides.

**The separating half, RDKit lists as unsupported.** The same Book names "component level grouping
requiring matches in different components, i.e. `(C).(C)`" among the SMARTS features it does not
have -- Daylight's third table row, named verbatim -- and `MolFromSmarts("(C).(N)")` duly returns
`None` (2025.09.3). RDKit does not need it in a
reaction, because `RunReactants` takes one molecule per reactant template -- the separation is
structural rather than a query constraint. vesma matches into one combined pot, so for vesma it
has to *be* a constraint, and that is the piece written here. For plain substructure matching there
is accordingly no second implementation to check against, and the semantics above are what the
tests assert.

On a reaction's right-hand side the grouping is a *postcondition*, not a match constraint: whether
a product comes apart is decided by the bonds the edit leaves behind, not by the query. An
application whose components disagree with what the template wrote is refused.

Enforced on CPU and Metal, both checked against each other. The CUDA kernels carry the same code
and are **not** validated: there is no CUDA device on the development machine and CI runs neither
GPU, so that path is written against the CPU reference and nothing has executed it.


## Rehash of SMARTS patterns

- v<n> (Total Valence): Specifies the total bond order of an atom (sum of bond orders, including implicit hydrogens).
  - Example: [Cv4] matches a carbon atom with a total bond order of 4 (e.g., in methane, ethane, benzene).
- X<n> (Total Connectivity): Specifies the total number of connections (explicitly bonded neighbors), excluding implicit hydrogens.
  - Example: [CX4] matches a carbon atom connected to 4 other heavy atoms.
- H<n> (Total H-Count): Specifies the total number of attached hydrogens (both implicit and explicit).
  - Example: [CH2] matches a carbon with exactly two hydrogens.
- h<n> (Implicit H-Count): Specifies the number of implicit hydrogens.
- D<n> (Explicit Degree): Specifies the number of explicit connections (neighbors).
- +<n> / -<n> (Formal Charge): Specifies the net formal charge of an atom.
  - Example: [N+] matches a positively charged nitrogen.

- Explicit vs. Implicit: In SMILES, [OH] is different from O. In SMARTS, [OH] specifically matches an oxygen with one explicit hydrogen, while O matches any oxygen.
- Valence Definition: The v primitive acts on the total bond order, which is often crucial when dealing with atoms that can have multiple valence states (e.g., S, P, N).
- Aromaticity: Lowercase letters (e.g., c, n) in SMARTS define aromatic atoms, which have implicit valence rules different from their aliphatic counterparts (capital letters).
- Wildcards: The * symbol matches any atom, while `[*,v3]` matches any atom with a valence of 3.

## Summary

- [Z] 128 bits (16 bytes) for atomic numbers 0..127.
- [A] 1 byte, bit0=aliphatic, bit1=aromatic.
- [C] 1 byte, buckets [-3-, -2, -1, 0, +1, +2, +3, +4+].
- [D, H, h, R, r, V, X, x] 1 byte each, buckets 0..7+.
- [M] atomic mass, parsed but unused.


## Parsing bracketed atom properties
SMARTS define the following levels of grammar:

1. Single property expression (let's call prop)
2. Possibly negated property expression (let's call neg_prop)
3. High-priority AND clause between neg_prop's. Separator is either "&"" or no separator at all. Let's call it high_and_clause.
4. OR clause between low_and_clause's. Separator is ",". Let's call it or_clause.
5. Low-priority AND clause between or_clause's. We can call it low_and_clause, or just plural props or props_expr.

Inside the brackets there is also optional atom mapping, colon followed by a number, like [C:1], but it comes last, after the properties expression. It can be part of bracketed expression parsing: [ + props_expr + atom_mapping? + ]

Single property is one of the following:

- Atomic number Z, prefixed by a "#"
- Atomic symbol (convertable to atomic number, plus aromaticity info, C vs c)
- Atomic mass M, no prefix, just a bare number
- A "Bucket": "D H h R r v X x - +", followed by an optional number
- A double charge symbol, either ++ or --

**Example**: High-priority AND can be either explicit "&" or implicit empty string.
For example, aliphatic carbon with mass 13 can be expressed either as [C13] or as [C&13], while [#6&13] have to have an explicit "&" between them.

Pyparsing: We can express high_and_clause as a list of neg_prop's, and just ignore the ampersand, like in the following: neg_prop + ZeroOrMore(Suppress(Optional("&")) + neg_prop)


## Reaction SMARTS

We can use the following:

  LHS>catalyst-etc>RHS

Or simply:

  LHS>>RHS

For retro reaction LHS describes the pattern in the product, and RHS prescribes how to break the product into precursors.

### Wildcard "\*"

Wildcards in LHS means any atom, wildcard in RHS means "keep whatever was in LHS for this atom map", i.e. `[C:1]>>[*:1]` means for atom map 1 copy whatever properties were in LHS (atom number Z=6 and, since capital letter C was used, aromaticity A=ALIPHATIC).

### LHS match vs RHS prescription

Hydrogen counts on LHS are used for matching. If hydrogen count appears on RHS, it instead prescribes protonation state to use in precursors. RHS can prescribe even Z: `[C:1]>>[N:1]` will change atom type from carbon to nitrogen for atom with atom map 1.

## "Easy" and "Difficult" properties

R, r, X, x, v are a little more difficult on re-percieved molecules. R and r require ring finding. `x` is total ring connections, also requires rings. X is total connectivity excluding implicit hydrogens (requires implicit hydrogens model). `v` total bond order, requires valence, aromaticity, hydrogens. H (total H-count) and h (implicit H-count) require hydrogen and valence model; only explicit hydrogens would have been easy, but they are not even in the SMARTS constraints.

Easy ones are: #Z (atomic number), C (formal charge, usually preserved), D (explicit connections, what was declared explicitly). We use those for robust hashes.
