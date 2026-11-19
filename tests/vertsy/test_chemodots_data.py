from __future__ import annotations

import bz2
import hashlib
from pathlib import Path

COMPRESSED_SHA256 = "f6c5d0fc4c24afa66f931c1d4c0636f437570a218eb10745863aafcc8bac12b0"
DECOMPRESSED_SHA256 = "5ece482b17fd0dd077cdce83bea9aa1cc124b28669630c01fc6d24a2268cf38b"
MOLECULE_ROWS = 988_112


def test_chemdots_enamine_bbs_file_matches_manifest(readonly_datadir: Path) -> None:
    stock_path = (
        readonly_datadir
        / "external"
        / "chemdots-stock"
        / "processed"
        / "chemdots_enamine_bbs.smi.bz2"
    )
    assert stock_path.exists()
    assert sha256_file(stock_path) == COMPRESSED_SHA256

    digest = hashlib.sha256()
    row_count = 0
    with bz2.open(stock_path, "rt", encoding="utf-8") as handle:
        first_line = handle.readline()
        assert first_line.strip() != "smiles"
        assert "\t" not in first_line
        digest.update(first_line.encode("utf-8"))
        row_count += 1

        for line in handle:
            assert "\t" not in line
            digest.update(line.encode("utf-8"))
            row_count += 1

    assert row_count == MOLECULE_ROWS
    assert digest.hexdigest() == DECOMPRESSED_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
