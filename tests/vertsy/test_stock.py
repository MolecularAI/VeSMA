from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vertsy import (
    cmol_rdkit_smiles_keyer,
    cmol_smiles_keyer,
    load_stock,
    load_stock_parallel,
    short_ring_wl_keyer,
)
from vertsy.stock import from_smiles


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("VESMA_CACHE", raising=False)
    monkeypatch.delenv("VESMA_N_JOBS", raising=False)


def test_load_stock_simple(tmp_path: Path):
    smi_file = tmp_path / "stock.smi"
    # Testing multiple formats (SMILES only, SMILES with ID)
    smi_file.write_text("C\nCC id1\nCCC\n")

    stock = load_stock(smi_file, cmol_smiles_keyer)

    assert len(stock) == 3
    # cmol_smiles_keyer writes every hydrogen explicitly, so that one molecule keys the
    # same however its hydrogens were spelled. Before that, methane keyed as "[C]" --
    # which is the SMILES for a bare carbon carrying none.
    assert "[CH4]" in stock
    assert "[CH3]-[CH3]" in stock
    assert "[CH2](-[CH3])-[CH3]" in stock


def test_load_stock_skips_invalid(tmp_path: Path):
    smi_file = tmp_path / "stock_invalid.smi"
    smi_file.write_text("C\nINVALID_SMILES\nCC\n")

    stock = load_stock(smi_file, cmol_smiles_keyer)

    assert len(stock) == 2
    assert "[CH4]" in stock
    assert "[CH3]-[CH3]" in stock


def test_load_stock_rdkit(tmp_path: Path):
    smi_file = tmp_path / "stock_rdkit.smi"
    smi_file.write_text("c1ccccc1\nC(C)C\n")

    stock = load_stock(smi_file, cmol_rdkit_smiles_keyer)

    # RDKit canonical SMILES for benzene is 'c1ccccc1', for propane is 'CCC'
    assert len(stock) == 2
    assert "c1ccccc1" in stock
    assert "CCC" in stock


def test_load_stock_short_ring_wl_tokens(tmp_path: Path):
    smi_file = tmp_path / "stock_short_ring_wl.smi"
    smi_file.write_text("C\nCC id1\nCCC\n")
    codec = short_ring_wl_keyer

    stock = load_stock(smi_file, codec)

    assert len(stock) == 3
    assert all(isinstance(token, tuple) and len(token) == 2 for token in stock)


def test_load_stock_cache_hit_returns_tokens_only(tmp_path: Path):
    smi_file = tmp_path / "stock_morgan_cache.smi"
    smi_file.write_text("C\nCC\n")
    first_codec = short_ring_wl_keyer
    expected_stock = load_stock(smi_file, first_codec)

    cached_codec = short_ring_wl_keyer

    def cache_miss(_):
        raise AssertionError("Expected load_stock to hit the joblib cache.")

    cached_codec = replace(cached_codec, from_smiles=cache_miss)
    observed_stock = load_stock(smi_file, cached_codec)

    assert observed_stock == expected_stock


def test_load_stock_caching(tmp_path: Path):
    smi_file = tmp_path / "cache_test.smi"
    smi_file.write_text("C\n")

    # Load once
    stock1 = load_stock(smi_file, cmol_smiles_keyer)

    # Track calls through an equivalent codec; load_stock's cache key uses the
    # codec name and version rather than object identity.
    call_count = 0
    original_fn = cmol_smiles_keyer.from_smiles

    def tracked_fn(smi):
        nonlocal call_count
        call_count += 1
        return original_fn(smi)

    tracked_codec = replace(cmol_smiles_keyer, from_smiles=tracked_fn)

    # Load again, should hit joblib cache and NOT call the processing function
    stock2 = load_stock(smi_file, tracked_codec)

    assert stock1 == stock2
    assert call_count == 0

    # Change file content, should re-run
    smi_file.write_text("CC\n")
    stock3 = load_stock(smi_file, tracked_codec)

    assert "[CH3]-[CH3]" in stock3
    assert call_count == 1


def test_from_smiles_cache_false_forces_reparse():
    smiles = ("C", "CC")
    expected = from_smiles(smiles, cmol_smiles_keyer, n_jobs=1)

    call_count = 0
    original_fn = cmol_smiles_keyer.from_smiles

    def tracked_fn(smi):
        nonlocal call_count
        call_count += 1
        return original_fn(smi)

    tracked_codec = replace(cmol_smiles_keyer, from_smiles=tracked_fn)
    observed = from_smiles(smiles, tracked_codec, n_jobs=1, cache=False)

    assert observed == expected
    assert call_count == 2


def test_from_smiles_cache_key_uses_exact_tuple_order():
    smiles = ("C", "CC")
    reversed_smiles = ("CC", "C")
    from_smiles(smiles, cmol_smiles_keyer, n_jobs=1)

    call_count = 0
    original_fn = cmol_smiles_keyer.from_smiles

    def tracked_fn(smi):
        nonlocal call_count
        call_count += 1
        return original_fn(smi)

    tracked_codec = replace(cmol_smiles_keyer, from_smiles=tracked_fn)

    assert from_smiles(smiles, tracked_codec, n_jobs=1) == {
        "[CH4]",
        "[CH3]-[CH3]",
    }
    assert call_count == 0

    assert from_smiles(reversed_smiles, tracked_codec, n_jobs=1) == {
        "[CH4]",
        "[CH3]-[CH3]",
    }
    assert call_count == 2


def test_load_stock_different_path_same_content(tmp_path: Path):
    file1 = tmp_path / "file1.smi"
    file2 = tmp_path / "file2.smi"
    content = "CCC\n"
    file1.write_text(content)
    file2.write_text(content)

    # Load file1
    load_stock(file1, cmol_smiles_keyer)

    # Patch to track
    call_count = 0
    original_fn = cmol_smiles_keyer.from_smiles

    def tracked_fn(smi):
        nonlocal call_count
        call_count += 1
        return original_fn(smi)

    tracked_codec = replace(cmol_smiles_keyer, from_smiles=tracked_fn)

    # Load file2 (different path, same content hash)
    stock2 = load_stock(file2, tracked_codec)

    assert "[CH2](-[CH3])-[CH3]" in stock2
    assert call_count == 0  # Should be a cache hit!


def test_load_stock_parallel_uses_same_cache(tmp_path: Path):
    smi_file = tmp_path / "parallel_cache_test.smi"
    smi_file.write_text("C\n")

    expected_stock = load_stock(smi_file, cmol_smiles_keyer)

    call_count = 0
    original_fn = cmol_smiles_keyer.from_smiles

    def tracked_fn(smi):
        nonlocal call_count
        call_count += 1
        return original_fn(smi)

    tracked_codec = replace(cmol_smiles_keyer, from_smiles=tracked_fn)
    observed_stock = load_stock_parallel(smi_file, tracked_codec)

    assert observed_stock == expected_stock
    assert call_count == 0
