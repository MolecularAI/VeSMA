"""Shared backend capacity limits.

Every number here sizes a buffer that a shader and its host must agree about exactly. A
disagreement is not a wrong answer that a parity test would catch -- it is a buffer read past its
end, or a row silently truncated. Two of them used to be declared bare in four modules each, so
they live here and the shaders are checked against them by `tests/vesma/test_shader_constants.py`,
which asks each `.mm` what it compiled rather than trusting a comment.

`GPU_` means both device backends. Where one backend's limit differs it carries that backend's
name, which is why the edited-atom caps are spelled `METAL_` and `CUDA_`.
"""

#: Atoms per molecule the dense backends accept, and so the width of one packed row: a packed
#: molecule is exactly as wide as the largest molecule the matcher takes, and there is no second
#: number. Modules that call this width a "bucket" alias it rather than restating 64.
GPU_MATCH_ATOM_CAP = 64

#: Atoms per constructed precursor component.
GPU_PRECURSOR_ATOM_CAP = 64

#: Property-mask words per atom in the packed layout -- the atom's declared properties as a
#: bitset, which is what makes a mask-engine match a word-wise AND.
ATOM_MASK_WORD_COUNT = 14

#: Atoms in a template's right-hand side, Metal. Larger than CUDA's deliberately: the Metal
#: kernel holds the edited pattern in device memory where CUDA holds it in a register-pressured
#: per-thread struct.
METAL_EDITED_ATOM_CAP = 256

#: Atoms in a template's right-hand side, CUDA.
CUDA_EDITED_ATOM_CAP = 128

#: Output component slots reserved per assignment in the retro kernel.
#:
#: Across the uspto42k corpus
#: the largest number of disconnected precursor components from one assignment is 6,
#: so 6 emits every component there with no truncation.
#: If an assignment ever splits into more,
#: the kernel keeps only the heaviest assignments
#: and sets a per-assignment overflow flag,
#: and the reaction matcher rejects that whole assignment --
#: a partial precursor set would give the search a false route.
MAX_COMPONENTS_PER_ASSIGNMENT = 16

__all__ = [
    "ATOM_MASK_WORD_COUNT",
    "CUDA_EDITED_ATOM_CAP",
    "GPU_MATCH_ATOM_CAP",
    "GPU_PRECURSOR_ATOM_CAP",
    "MAX_COMPONENTS_PER_ASSIGNMENT",
    "METAL_EDITED_ATOM_CAP",
]
