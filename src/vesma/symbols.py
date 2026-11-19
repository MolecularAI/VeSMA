from __future__ import annotations

# Position is the atomic number, so this must stay dense and in order: index 0 is the
# ``*`` wildcard and index n is element n. Complete through oganesson, because the table
# is also what the parser matches symbols against -- stopping early does not merely omit
# an element, it makes its symbol parse as something else. `[Ra]` read as a ring-membership
# primitive plus a wildcard rather than as radium, which is the failure this length
# prevents. Declaring a symbol is not declaring a chemistry: an element with no entry in
# ``configurations.py`` still refuses with ``undeclared_configuration``.
PERIODIC_SYMBOLS = (
    "* H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
)
Z_TO_ELEMENT: dict[int, str] = dict(enumerate(PERIODIC_SYMBOLS.split()))
ELEMENT_TO_Z: dict[str, int] = {
    symbol: idx for idx, symbol in enumerate(PERIODIC_SYMBOLS.split())
}

# OpenSMILES 1.0 section 2.2: b, c, n, o, p, s, se, as.
AROMATIC_SYMBOLS = "b c n o p s se as"
# ``capitalize`` rather than ``upper``: the two-letter aromatics must map to ``Se`` and
# ``As``, the real element symbols. ``upper`` produced ``SE``, which matches no element,
# so the writer never recognised aromatic selenium and emitted it as uppercase ``[Se]``
# inside a lowercase ring.
AROMATIC_UNBRACKET = {
    symbol: symbol.capitalize() for symbol in AROMATIC_SYMBOLS.split()
}
UNBRACKET_SYMBOLS = "Cl Br B C N O P S F I b c n o p s a *"
ALL_ATOM_SYMBOLS = f"{PERIODIC_SYMBOLS} {AROMATIC_SYMBOLS} a A"
