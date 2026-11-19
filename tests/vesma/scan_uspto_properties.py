import csv
import gzip
import re
from collections import Counter

from tests.datadirs import external_data_root


def main():
    template_file = (
        external_data_root()
        / "external"
        / "paroutes"
        / "processed"
        / "uspto_unique_templates.csv.gz"
    )

    # We want to match:
    # R (ring membership): R or R<num>
    # r (ring size): r or r<num>
    # x (ring connectivity): x or x<num>
    # v (valence): v or v<num>
    # X (connectivity): X or X<num>
    # But only inside brackets [...] of SMARTS.

    # Regex to find all bracketed expressions
    bracket_pat = re.compile(r"\[([^\]]+)\]")

    # Patterns for properties (case-sensitive or not? SMARTS properties are case-sensitive:
    # R, r, x, v, X are case-sensitive. Let's make sure we match them as words or separators.
    # Inside brackets, properties are typically separated by ; or & or , or just concatenated,
    # e.g., [#6;R1], [C;r5], [x2], [v3], [X3].
    # Let's write a regex that matches:
    # R\d* (as a separate component inside brackets, i.e., not preceded or followed by other letters/digits that make it something else)
    # Actually, let's just inspect the characters inside each bracket.
    # To be extremely precise, let's tokenize the brackets or match:
    # (?<![a-zA-Z])R(\d*)(?![a-zA-Z])
    # (?<![a-zA-Z])r(\d*)(?![a-zA-Z])
    # (?<![a-zA-Z])x(\d*)(?![a-zA-Z])
    # (?<![a-zA-Z])v(\d*)(?![a-zA-Z])
    # (?<![a-zA-Z])X(\d*)(?![a-zA-Z])
    # Note that element symbols can also be letters, but inside brackets elements are like C, N, O, F, Cl, Br, I, etc.
    # Or specified as [#6]. A property token is usually R<num>, r<num>, x<num>, v<num>, X<num>, etc.
    # Sometimes it can be just R, r, x, v, X.
    # Let's count how many times each matches in the entire dataset.

    patterns = {
        "R": re.compile(r"\b[R]\d*\b"),
        "r": re.compile(r"\b[r]\d*\b"),
        "x": re.compile(r"\b[x]\d*\b"),
        "v": re.compile(r"\b[v]\d*\b"),
        "X": re.compile(r"\b[X]\d*\b"),
    }

    counts = Counter()
    total_templates = 0

    # Let's also see what unique matches we get, to ensure we don't match elements or other things incorrectly.
    matches_found = {k: Counter() for k in patterns}

    with gzip.open(template_file, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tmpl = row["retro_template"]
            total_templates += 1

            # Find all brackets
            brackets = bracket_pat.findall(tmpl)

            # For each bracket, check for our properties
            for b in brackets:
                # To avoid matching elements like Ru, Rh, Re (which start with R)
                # or Xe (starts with X), let's split the bracket content by common SMARTS operators:
                # semicolon (;), ampersand (&), comma (,), and map colon (:) to handle atom maps.
                # E.g., 'C;H0;D3;+0:1' -> ['C', 'H0', 'D3', '+0', '1']
                tokens = re.split(r"[;&,:]", b)
                for tok in tokens:
                    tok = tok.strip()
                    # Check if token matches our properties
                    for prop_name, pat in patterns.items():
                        m = pat.match(tok)
                        if m:
                            counts[prop_name] += 1
                            matches_found[prop_name][tok] += 1

    print(f"Total templates analyzed: {total_templates}")
    print("\nCounts of SMARTS properties in USPTO 50k templates:")
    for prop_name in sorted(patterns.keys()):
        print(f"  Property '{prop_name}': {counts[prop_name]} occurrences")
        print(f"    Top matches: {dict(matches_found[prop_name].most_common(10))}")


if __name__ == "__main__":
    main()
