from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, cast

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from vesma.keyer import Keyer

from .core import K
from .heuristics import Heuristic
from .search import MoleculeBatch

__all__ = [
    "FINGERPRINTS",
    "atom_pair_generator",
    "ecfp4_matrix",
    "ecfp4_matrix_from_smiles",
    "colliding_stock_neighbours",
    "fingerprint_matrix",
    "morgan_generator",
    "parsed_smiles",
    "stock_of",
    "stock_similarity",
]

ECFP4_RADIUS = 2
ECFP4_BITS = 2048


# Four views of a molecule, each a different answer to "what counts as a feature". They are known to
# correlate — that is the finding this exists to exploit rather than to deny, since
# :func:`vertsy.heuristics.strongest` combines estimates by veto rather than by vote, and a view
# only has to disagree *sometimes* to be worth its matmul.
#
#   ecfp4     circular environments up to radius 2. Substructure identity; the incumbent.
#   path      RDKit topological paths. Connectivity and shape rather than local environment.
#   atompair  hashed atom pairs with topological distance. Long-range relationships the other
#             two cannot see, since both are local by construction.
#   maccs     166 hand-drawn structural keys. Fixed, tiny, and the only one whose features a
#             chemist named — so its disagreements with the others are the interpretable ones.
#
# `n_bits` is honoured where the fingerprint is hashed and ignored for MACCS, which is 167 bits by
# definition. A caller mixing them must not assume a shared width.
#
# The ``-count`` views keep how many times each feature occurs instead of whether it occurs at all,
# which is the finer question: a molecule carrying three of a motif and one carrying four are the
# same point to a bit vector. They are scored with the *continuous* Tanimoto,
# ``a.b / (|a|^2 + |b|^2 - a.b)``, which is all dot products and so stays one BLAS call — the
# MinMax form ``sum min / sum max`` is the usual count generalisation and cannot be a matrix
# product at all, which would cost the whole reason this heuristic is affordable. On 0/1 input the
# continuous form *is* the binary one, since a bit equals its own square.
def _counts(vector) -> dict[int, int]:
    """Index -> count, for either a bit vector or a count vector."""
    if hasattr(vector, "GetNonzeroElements"):
        return dict(vector.GetNonzeroElements())
    return {int(index): 1 for index in vector.GetOnBits()}


@lru_cache(maxsize=8)
def morgan_generator(n_bits: int) -> Any:
    """The reusable ECFP4 generator for one fingerprint width.

    A generator rather than ``rdMolDescriptors.GetMorganFingerprintAsBitVect``, which RDKit has
    deprecated in favour of exactly this and which rebuilds the generator per molecule. Cached per
    width because the caller fingerprints a whole stock: building it once is the point.
    """
    return rdFingerprintGenerator.GetMorganGenerator(radius=ECFP4_RADIUS, fpSize=n_bits)


@lru_cache(maxsize=8)
def atom_pair_generator(n_bits: int) -> Any:
    """The reusable hashed atom-pair generator for one fingerprint width."""
    return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)


FINGERPRINTS: dict[str, Callable[[Chem.Mol, int], Any]] = {
    "ecfp4": lambda mol, n_bits: morgan_generator(n_bits).GetFingerprint(mol),
    "ecfp4-count": lambda mol, n_bits: morgan_generator(n_bits).GetCountFingerprint(
        mol
    ),
    "path": lambda mol, n_bits: Chem.RDKFingerprint(mol, fpSize=n_bits),
    "atompair": lambda mol, n_bits: atom_pair_generator(n_bits).GetFingerprint(mol),
    "atompair-count": lambda mol, n_bits: atom_pair_generator(
        n_bits
    ).GetCountFingerprint(mol),
    "maccs": lambda mol, n_bits: rdMolDescriptors.GetMACCSKeysFingerprint(mol),
}
MACCS_BITS = 167


def ecfp4_matrix(molecules: Sequence[Chem.Mol], n_bits: int = ECFP4_BITS) -> np.ndarray:
    """Unpacked ECFP4 bit matrix, one ``float32`` row per molecule.

    Unpacked and float rather than packed bits because the only consumer multiplies it (see
    :func:`stock_similarity`), and BLAS wants floats. Set bits are written by index, which
    keeps the row shape fixed no matter what RDKit returns.

    Molecules rather than SMILES: the search hands over the very objects its matcher produced,
    and parsing was two thirds of this function's cost when it took strings. Use
    :func:`ecfp4_matrix_from_smiles` for stock, which really does arrive as text.
    """
    return fingerprint_matrix(molecules, "ecfp4", n_bits)


def fingerprint_matrix(
    molecules: Sequence[Chem.Mol], kind: str = "ecfp4", n_bits: int = ECFP4_BITS
) -> np.ndarray:
    """:func:`ecfp4_matrix` over any view in :data:`FINGERPRINTS`, same shape and dtype.

    One function rather than four because every coefficient downstream is three sums of a bit
    matrix, so the *only* thing a fingerprint changes is which bits are set. Keeping that the single
    difference is what makes several views combinable at all.
    """
    if kind not in FINGERPRINTS:
        raise ValueError(
            f"unknown fingerprint {kind!r}; expected one of {sorted(FINGERPRINTS)}."
        )
    width = MACCS_BITS if kind == "maccs" else n_bits
    matrix = np.zeros((len(molecules), width), dtype=np.float32)
    build = FINGERPRINTS[kind]
    for index, mol in enumerate(molecules):
        if mol is None:
            raise ValueError(f"molecule at index {index} is None.")
        counted = _counts(build(mol, n_bits))
        if counted:
            matrix[index, list(counted)] = list(counted.values())
    return matrix


def parsed_smiles(smiles: Sequence[str]) -> list[Chem.Mol]:
    """Parse or raise, saying which row failed. The stock is the one input that arrives as text."""
    molecules = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"cannot parse SMILES at index {index}: {smi!r}")
        molecules.append(mol)
    return molecules


def ecfp4_matrix_from_smiles(
    smiles: Sequence[str], n_bits: int = ECFP4_BITS
) -> np.ndarray:
    """:func:`ecfp4_matrix` for molecules that only exist as text, such as a stock file."""
    return ecfp4_matrix(parsed_smiles(smiles), n_bits)


def stock_similarity(
    stock_smiles: Sequence[str],
    *,
    weight: float,
    fingerprint: str = "ecfp4",
    neighbours: int = 1,
    idf: bool = False,
    n_bits: int = ECFP4_BITS,
) -> Heuristic[Chem.Mol]:
    """Estimate remaining reactions as ``weight *`` the Jaccard distance to the nearest stock.

    A molecule that *is* in the stock scores 0 and one sharing nothing with it scores
    ``weight``, so the estimate vanishes at the goal as :class:`Heuristic` requires. The
    chemistry it encodes is only a guess — fingerprint distance is not synthetic distance —
    but it is a guess that decreases as a molecule decomposes toward buyable fragments.

    Jaccard, not Hamming, and 2048 bits, not 128, for the same reason: both alternatives
    smuggle molecule size back in. Hamming counts differing bits, so a large molecule is far
    from everything regardless of how buyable its pieces are (measured on PaRoutes n1:
    Hamming correlates +0.80 with heavy-atom count, Jaccard at 2048 bits +0.06). A heuristic
    that is really a size proxy just re-derives "big molecules are hard", which the search
    already knows from the reactions it has committed to.

    ``weight`` is required and deliberately has no default, because a dimensionless distance has no
    natural cost and every default is a guess wearing a convention's clothes. It converts Jaccard
    into whatever unit ``rule_cost`` charges, and only the **ratio** of the two decides the search:
    scaling both changes nothing. **Start about 10x the step cost** — with a step of 0.105 that is
    ``weight = 1``, which is where the measurement below lands — then read the ``drop`` column of
    :func:`vertsy.effort.per_reaction`, which reports how far this estimate actually falls per
    reaction, and move the ratio toward what it says.

    Where the 10x comes from, two ways. Measured against PaRoutes n1 reference depths, the mean
    Jaccard distance falls about 0.04 per reaction over the interior of a route; measured from a
    run's own picks it falls 0.08 to 0.12 per level, stably enough across depths and effort shapes
    to be a property of the stock rather than of the sample.

    And the search reads it back. Sixteen seeded n1 targets per reference depth, ``weight = 1``,
    32x8, against a step-1.0 control on the same targets: **34/48 solved at 1.0 and 44/48 at 0.1**,
    the margin widening with depth (12→14, 12→15, 10→15 at depths 5, 6, 7) because a unit step's
    overcharge compounds with how deep the answer sits. Cost: at most 0.2 reactions of route length.
    Below about 0.05 the solve rate holds but quality starts being paid for in earnest.

    At ``weight <= min rule_cost`` the estimate never exceeds the truth — any non-stock molecule
    needs at least one reaction — and AO* returns the cheapest route within its depth budget. Above
    it you are running weighted A*: usually faster, no optimality claim. The 10x hint is on the
    inadmissible side of that line by an order of magnitude, which is the trade it is recommending.

    ``neighbours`` averages the *k* nearest stock molecules instead of taking the single nearest,
    and ``idf`` weights each bit by ``log(N / df)`` over the catalogue so a bit set in most of the
    stock stops counting as much as a rare one. Both rank **the same as the incumbent to
    within noise** against reference depths on n1 (rho +0.41 to +0.46, standard error 0.027 — do not
    read the ordering of those numbers), and both make the estimate almost injective: 1.00 to 1.01
    open leaves per distinct value against the incumbent's 2.28.

    **Do not adopt either for that injectivity.** It was the reason they were written and it turned
    out not to be a reason at all. The estimate's ties are not the search's — selection sorts on
    ``context + inside``, and the context separates nearly everything the estimate ties. Measured
    over 1,344 real passes, the tie at the selection boundary is a median of **2 leaves out of a
    4,949-leaf frontier**: common (56% of passes) but tiny, so a finer estimate moves at most one
    slot in sixteen, between two molecules that scored alike. And it moves nothing: ranking by ecfp4
    with ecfp4-count breaking its exact ties changed **zero** targets over 64 searches, gained or
    lost, routes included. A no-op rather than a wash.

    They do **not** compose: ``idf`` widens the gradient (mean estimate from depth 1 to 6 spans 0.212
    against the incumbent's 0.207) while averaging compresses it (0.152 at k=3, 0.098 at k=10), and
    the compression wins. Take one. Larger ``neighbours`` is also not better — rho falls to +0.400 at
    k=10 and +0.321 at k=50 as the catalogue's bulk drowns the nearest match.

    ``fingerprint`` selects the view — see :data:`FINGERPRINTS`. Several of these combined by
    :func:`vertsy.heuristics.strongest` stay admissible and dominate each one alone, which is the
    supported way to use more than one: a molecule then scores low only if every view agrees it is
    close. Give each its own ``weight``, since MACCS at 167 bits and ECFP4 at 2048 do not produce
    distances on the same scale even though both live in ``[0, 1]``.

    Scoring is exact against the whole stock and runs as one BLAS matmul, since the
    intersection of two bit sets is their dot product — 0.13-1.7 ms per query against the
    13.4k-molecule n1 stock, cheaper per query the bigger the batch, which suits a caller
    that expands in batches. The stock matrix costs ``4 * len(stock) * n_bits`` bytes
    (110 MB for n1); pack it and use ``np.bitwise_count`` to trade ~3x time for ~32x memory,
    or reach for an ANN index if the stock ever reaches catalogue scale.
    """
    if not len(stock_smiles):
        raise ValueError("stock_smiles must be non-empty to score similarity to it.")
    if neighbours < 1:
        raise ValueError(f"neighbours must be at least one, got {neighbours!r}.")
    stock = fingerprint_matrix(parsed_smiles(stock_smiles), fingerprint, n_bits)
    # Weighting both sides by sqrt(w) makes the weighted Tanimoto fall out of the same three sums,
    # so the whole cost of `idf` is one precomputed vector and no change to the matmul at all.
    scale = None
    if idf:
        document = (stock > 0).sum(axis=0)
        scale = np.sqrt(np.log(len(stock) / (1.0 + document)).clip(min=0.0)).astype(
            np.float32
        )
        stock = stock * scale
    take = min(neighbours, len(stock))
    # `sum` would be the bit count; the square is the same number on 0/1 input and the right
    # one on counts, so this single expression serves both without a branch.
    stock_norm = (stock * stock).sum(axis=1)

    def similarity(molecules: MoleculeBatch[Chem.Mol]) -> np.ndarray:
        if not len(molecules):
            return np.zeros(0, dtype=np.float32)
        # Iterability is this heuristic's one requirement of the batch, and the cast is where
        # it is asserted: RDKit fingerprints need real `Mol` objects, so this pairs with
        # `object_mol_array` and a packed device batch would fail here — loudly, not silently.
        query = fingerprint_matrix(
            cast("Sequence[Chem.Mol]", molecules), fingerprint, n_bits
        )
        if scale is not None:
            query = query * scale
        intersection = query @ stock.T
        union = (
            (query * query).sum(axis=1)[:, None] + stock_norm[None, :] - intersection
        )
        # A molecule with no set bits meets an empty union; its intersection is 0 too, so
        # dividing by 1 yields the honest answer (shares nothing) instead of a nan.
        tanimoto = intersection / np.maximum(union, 1.0)
        if take == 1:
            nearest = tanimoto.max(axis=1)
        else:
            # `partition` rather than a sort: the k best are wanted, their order is not.
            nearest = -np.partition(-tanimoto, take - 1, axis=1)[:, :take].mean(axis=1)
        return (weight * (1.0 - nearest)).astype(np.float32)

    return similarity


def stock_of(
    smiles: Sequence[str],
    keyer: Keyer[Chem.Mol, K],
    *,
    weight: float,
    fingerprint: str = "ecfp4",
    neighbours: int = 1,
    idf: bool = False,
    n_bits: int = ECFP4_BITS,
) -> tuple[frozenset[K], Heuristic[Chem.Mol]]:
    """The keys a search calls buyable and the heuristic that measures distance to them, together.

    A search takes ``stock`` and ``heuristic`` as separate arguments, and nothing checks that
    they describe the same catalogue — one can be keyed from one file while the other is
    fingerprinted from another, and the only symptom is a molecule the estimate calls finished
    that solvedness never accepts. Building both from one list of SMILES removes that whole
    class of disagreement: same molecules, same order, one parse.

    **Both derive from the SMILES rows, and never a key from a fingerprint or back.** ``K`` is
    generic and a key need not be invertible — canonical SMILES happens to be, the short-ring WL
    key vesma uses is a hash — so the shared rows are the only thing the two can agree on. Keys
    go through ``keyer.from_smiles``, the same call :func:`vertsy.stock.load_stock` makes, so a
    stock built here is bit-identical to one loaded from the same file.

    What this cannot fix, and should not be read as fixing: key equality and fingerprint
    equality remain different relations. A fingerprint is a *partition* of molecule space, so
    ``h = 0`` means "in the same class as something buyable", which is strictly weaker than
    "is buyable" — binary ECFP4 puts every long unbranched chain of the same ends in one class.
    Unifying the inputs is what makes that residue interpretable: once both stocks are the same
    molecules, a zero on a non-stock molecule can only be the fingerprint, and
    :func:`vertsy.audit.auditing_fold`'s report becomes a measurement rather than a puzzle.
    """
    keys = frozenset(keyer.from_smiles(smi) for smi in smiles)
    heuristic = stock_similarity(
        smiles,
        weight=weight,
        fingerprint=fingerprint,
        neighbours=neighbours,
        idf=idf,
        n_bits=n_bits,
    )
    return keys, heuristic


def colliding_stock_neighbours(
    molecules: Sequence[Chem.Mol],
    stock: Sequence[str],
    *,
    fingerprint: str = "ecfp4",
    n_bits: int = ECFP4_BITS,
) -> list[tuple[int, int, float]]:
    """``(molecule index, stock index, similarity)`` for molecules indistinguishable from stock.

    The counterpart to :func:`vertsy.audit.auditing_fold`'s zero report, and the reason both
    exist. The search knows which molecules are buyable and can therefore *count* the ones the
    heuristic prices at zero without owning them; it never sees a molecule, so it can never say
    which. This has the molecules and the matched row, so it can name the pair — and naming is
    what turns "127 of 128 picks estimated at zero" into "binary ECFP4 cannot count repeated
    identical environments, so past five carbons chain length is invisible":

        ``C=CCCCCCCO`` (9 heavy atoms) and ``C=CCCCCCCCCCO`` (12) share all sixteen bits.

    A diagnostic, so it recomputes rather than reaching into the heuristic's closure, and it is
    dense in the stock — call it on the molecules a report has already narrowed to.
    """
    if not len(molecules):
        return []
    stock_matrix = fingerprint_matrix(parsed_smiles(stock), fingerprint, n_bits)
    query = fingerprint_matrix(molecules, fingerprint, n_bits)
    intersection = query @ stock_matrix.T
    union = (
        (query * query).sum(axis=1)[:, None]
        + (stock_matrix * stock_matrix).sum(axis=1)[None, :]
        - intersection
    )
    tanimoto = intersection / np.maximum(union, 1.0)
    nearest = tanimoto.argmax(axis=1)
    best = tanimoto.max(axis=1)
    return [
        (index, int(nearest[index]), float(best[index]))
        for index in range(len(molecules))
        if best[index] >= 1.0 - 1e-9
    ]
