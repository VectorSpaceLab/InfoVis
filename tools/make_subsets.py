"""Materialize the CPT replay subsets from their id lists.

The data release ships each sampled subset as `cpt/subsets/<name>.ids`, one
record id per line. A record id is `<pool file without .jsonl>/<line index>`,
e.g. `cpt/t2i/academic_long/000123456`. This script writes
`cpt/subsets/<name>.jsonl` next to each id list.

    python tools/make_subsets.py --data_root data/InfoVis-Data
"""

import argparse
import collections
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()

    for ids_path in sorted(glob.glob(os.path.join(args.data_root, "cpt/subsets/*.ids"))):
        out_path = ids_path[:-4] + ".jsonl"
        if os.path.exists(out_path):
            continue
        wanted = collections.defaultdict(set)
        order = []
        with open(ids_path) as f:
            for line in f:
                pool, idx = line.strip().rsplit("/", 1)
                wanted[pool].add(int(idx))
                order.append((pool, int(idx)))
        rows = {}
        for pool, idxs in wanted.items():
            with open(os.path.join(args.data_root, pool + ".jsonl")) as f:
                for i, line in enumerate(f):
                    if i in idxs:
                        rows[(pool, i)] = line
        with open(out_path, "w") as f:
            for key in order:
                f.write(rows[key])
        print(f"{out_path}: {len(order)} records")


if __name__ == "__main__":
    main()
