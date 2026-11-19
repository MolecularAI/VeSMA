from __future__ import annotations

import importlib


def test_src_layout_imports_work() -> None:
    assert importlib.import_module("vesma").__name__ == "vesma"
    assert importlib.import_module("vertsy").__name__ == "vertsy"
