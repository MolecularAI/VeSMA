import argparse
import cProfile
import gzip
import pstats
import time
from pathlib import Path

from tests.datadirs import external_data_root
from vesma.reaction import parse_reaction_smarts


def run_benchmark(dataset_path: Path, limit: int = 1000, do_profile: bool = True):
    templates = []
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    with gzip.open(dataset_path, "rt") as f:
        # Skip header
        f.readline()
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.split("\t")
            if len(parts) > 1:
                templates.append(parts[1])  # retro_template column

    print(f"Loaded {len(templates)} templates.")

    start_time = time.perf_counter()

    profiler = None
    if do_profile:
        profiler = cProfile.Profile()
        profiler.enable()

    count = 0
    errors = 0
    for template in templates:
        try:
            parse_reaction_smarts(template)
            count += 1
        except Exception:
            errors += 1

    if profiler:
        profiler.disable()

    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Parsed {count} templates with {errors} errors in {duration:.2f} seconds.")
    if len(templates) > 0:
        print(f"Average time per template: {duration / len(templates) * 1000:.3f} ms")

    if profiler:
        stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
        stats.print_stats(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()

    dataset_path = (
        external_data_root()
        / "external"
        / "paroutes"
        / "processed"
        / "uspto_unique_templates.csv.gz"
    )
    run_benchmark(dataset_path, limit=args.limit, do_profile=not args.no_profile)
