import argparse
import gzip

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, FilterCatalog


def process_batch(smiles_batch, rules_smarts):
    # Re-parse rules in the worker to avoid pickle issues with complex RDKit objects
    params = FilterCatalog.FilterCatalogParams()
    catalog = FilterCatalog.FilterCatalog(params)
    rxn_dict = {}
    for i, smarts in enumerate(rules_smarts):
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn.GetNumReactantTemplates() > 0:
                template = rxn.GetReactantTemplate(0)
                smarts_filter = FilterCatalog.SmartsMatcher(template)
                entry = FilterCatalog.FilterCatalogEntry(f"Rule_{i}", smarts_filter)
                catalog.AddEntry(entry)
                rxn_dict[f"Rule_{i}"] = rxn
        except Exception:
            pass

    f1_set = set()
    for smi in smiles_batch:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        matches = catalog.GetMatches(mol)
        for match in matches:
            rxn = rxn_dict[match.GetDescription()]
            try:
                products = rxn.RunReactants((mol,))
                for prod_tuple in products:
                    for p in prod_tuple:
                        try:
                            Chem.SanitizeMol(p)
                            p_smi = Chem.MolToSmiles(p, isomericSmiles=False)
                            f1_set.add(p_smi)
                        except Exception:
                            pass
            except Exception:
                pass
    return f1_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--f0-out", required=True)
    parser.add_argument("--f1-out", required=True)
    parser.add_argument("--f01-out", required=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    print(f"Loading targets from {args.targets}...")
    with gzip.open(args.targets, "rt") as f:
        f0_smiles = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(f0_smiles)} targets (f0).")

    with gzip.open(args.f0_out, "wt") as f:
        for smi in sorted(f0_smiles):
            f.write(smi + "\n")

    print(f"Loading rules from {args.rules}...")
    lib_df = pd.read_csv(args.rules, sep="\t")
    rules_smarts = lib_df["retro_template"].dropna().tolist()
    print(f"Loaded {len(rules_smarts)} rule SMARTS.")

    f0_list = list(f0_smiles)
    # Split into batches
    num_jobs = args.n_jobs if args.n_jobs > 0 else 32
    batch_size = max(1, len(f0_list) // num_jobs)
    batches = [f0_list[i : i + batch_size] for i in range(0, len(f0_list), batch_size)]

    print(f"Processing in {len(batches)} batches using {args.n_jobs} jobs...")
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_batch)(batch, rules_smarts) for batch in batches
    )

    f1_smiles = set()
    for res in results:
        f1_smiles.update(res)

    print(f"Found {len(f1_smiles)} unique f1 precursors.")

    with gzip.open(args.f1_out, "wt") as f:
        for smi in sorted(f1_smiles):
            f.write(smi + "\n")

    f01_smiles = f0_smiles | f1_smiles
    print(f"Total f01 molecules: {len(f01_smiles)}")

    with gzip.open(args.f01_out, "wt") as f:
        for smi in sorted(f01_smiles):
            f.write(smi + "\n")


if __name__ == "__main__":
    main()
