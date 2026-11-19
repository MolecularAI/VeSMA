from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts

RAW_TEMPLATE_COLUMNS = (
    "template_code",
    "retro_template",
    "template_hash",
    "classification",
    "library_occurence",
)
PATTERN_COLUMNS = (
    "template_code",
    "template_hash",
    "classification",
    "library_occurence",
    "retro_template",
    "lhs_smarts",
    "lhs_atom_count",
    "hit_class",
)
MOLECULE_COLUMNS = ("chembl_id", "smiles", "atom_count")
SELECTED_PATTERN_CODES = (
    "0",
    "1",
    "6",
    "7",
    "9",
    "10",
    "15",
    "16",
    "2",
    "3",
    "4",
    "5",
    "8",
    "11",
    "12",
    "13",
)
POSITIVE_PATTERN_CODES = frozenset({"0", "1", "6", "7", "9", "10", "15", "16"})
SELECTED_MOLECULE_IDS = (
    "394110",
    "724325",
    "747824",
    "1168162",
    "897356",
    "153180",
    "395578",
    "1064159",
    "396137",
    "399330",
    "889479",
    "1239557",
    "415958",
    "415951",
    "800717",
    "415956",
    "985337",
    "399327",
    "395549",
    "395572",
    "395566",
    "1160997",
    "414449",
    "43997",
    "472856",
    "398014",
    "912641",
    "356194",
    "403100",
    "395553",
    "395573",
    "415914",
    "395571",
    "399328",
    "395570",
    "892149",
    "963311",
    "413972",
    "1104784",
    "395554",
    "403098",
    "1408234",
    "595784",
    "769685",
    "303916",
    "415915",
    "552578",
    "21376",
    "501431",
    "860715",
    "205136",
    "414440",
    "759689",
    "545488",
    "430755",
    "399333",
    "395563",
    "769657",
    "414436",
    "996967",
    "583638",
    "553286",
    "266176",
    "396833",
    "250427",
    "495236",
    "949486",
    "551540",
    "541769",
    "682874",
    "395557",
    "413967",
    "377672",
    "399317",
    "340296",
    "414448",
    "804521",
    "1119867",
    "395568",
    "316123",
    "66098",
    "249610",
    "1010724",
    "867896",
    "399460",
    "396964",
    "430715",
    "953362",
    "439373",
    "75368",
    "1443840",
    "278864",
    "358704",
    "329083",
    "854508",
    "439522",
    "404763",
    "403458",
    "46407",
    "969423",
    "337547",
    "663883",
    "593868",
    "20536",
    "403730",
    "519141",
    "470211",
    "860239",
    "969698",
    "1211127",
    "252642",
    "810998",
    "415916",
    "395567",
    "741937",
    "1302341",
    "149596",
    "395556",
    "878718",
    "747129",
    "1221747",
    "1048218",
    "395547",
    "790973",
    "441610",
    "379345",
    "976161",
    "439575",
)


def main() -> None:
    args = _parse_args()
    dataset_dir = args.dataset_dir.resolve()

    external_dir = dataset_dir.parent.parent
    template_rows = _template_rows_by_code(
        external_dir / "paroutes" / "processed" / "uspto_unique_templates.csv.gz"
    )
    smiles_by_id = _smiles_by_id(
        external_dir / "chembl" / "raw" / "chembl16_50K_sss.txt.gz"
    )

    _write_tsv(
        dataset_dir / "patterns.tsv",
        PATTERN_COLUMNS,
        [_pattern_row(code, template_rows[code]) for code in SELECTED_PATTERN_CODES],
    )
    _write_tsv(
        dataset_dir / "molecules.tsv",
        MOLECULE_COLUMNS,
        [
            _molecule_row(chembl_id, smiles_by_id[chembl_id])
            for chembl_id in SELECTED_MOLECULE_IDS
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
    )
    return parser.parse_args()


def _template_rows_by_code(path: Path) -> dict[str, dict[str, str]]:
    with gzip.open(
        path, "rt", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != RAW_TEMPLATE_COLUMNS:
            raise ValueError(f"{path} has unexpected columns: {fieldnames!r}")
        rows = {row["template_code"]: dict(row) for row in reader}

    missing = [code for code in SELECTED_PATTERN_CODES if code not in rows]
    if missing:
        raise ValueError(f"Missing template rows in {path}: {missing!r}")
    return rows


def _smiles_by_id(path: Path) -> dict[str, str]:
    smiles_by_id: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            chembl_id, smiles = stripped.split(maxsplit=1)
            smiles_by_id[chembl_id] = smiles

    missing = [
        chembl_id
        for chembl_id in SELECTED_MOLECULE_IDS
        if chembl_id not in smiles_by_id
    ]
    if missing:
        raise ValueError(f"Missing molecule rows in {path}: {missing!r}")
    return smiles_by_id


def _pattern_row(template_code: str, raw_row: dict[str, str]) -> dict[str, str | int]:
    retro_template = raw_row["retro_template"]
    lhs_smarts = retro_template.split(">>", 1)[0].strip()
    try:
        lhs_query = gmol_from_smarts(lhs_smarts)[0]
    except SmartsError as exc:
        raise ValueError(
            f"Selected template {template_code} no longer parses on the benchmark LHS."
        ) from exc

    return {
        "template_code": template_code,
        "template_hash": raw_row["template_hash"],
        "classification": raw_row["classification"],
        "library_occurence": int(raw_row["library_occurence"]),
        "retro_template": retro_template,
        "lhs_smarts": lhs_smarts,
        "lhs_atom_count": lhs_query.atom_count,
        "hit_class": (
            "positive" if template_code in POSITIVE_PATTERN_CODES else "negative"
        ),
    }


def _molecule_row(chembl_id: str, smiles: str) -> dict[str, str | int]:
    try:
        molecule = gmol_from_cmol(cmol_from_smiles(smiles)[0])
    except SmartsError as exc:
        raise ValueError(
            f"Selected molecule {chembl_id} no longer parses with the current Vesma code."
        ) from exc
    return {
        "chembl_id": chembl_id,
        "smiles": smiles,
        "atom_count": molecule.atom_count,
    }


def _write_tsv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, str | int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
