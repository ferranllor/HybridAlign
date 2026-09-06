#!/usr/bin/env python3
"""
Structure of the input graphs, which is the one thing in the report that does not depend on the
machine: how many nodes sit at every topological level, and how many bases they carry.

    python3 tools/report_graph_stats.py                          # default datasets
    python3 tools/report_graph_stats.py -d 150_10 brca2_150
    python3 tools/report_graph_stats.py -o outputs/report

Writes <out>/graph_levels.csv with one row per (dataset, level):

    dataset,level,width,bases,maxlen

"width" is how many nodes share that level and is therefore how much node parallelism a level
offers; "bases" is the sum of their sequence lengths, i.e. the DP work of the level divided by the
query length. Both are what decides whether a level is worth a kernel launch at all, so the whole
GPU/CPU split argument of the report is read off this file.

The level of a node is its longest path distance from a source, which is exactly what
sort_graph_topologically() in main.c computes, so these numbers are the ones bin/main schedules on.

Run from the repository root.
"""

import argparse
import collections
import csv
import os
import sys

DEFAULT_DATASETS = ["150_10", "brca2_150"]


def graph_path(dataset):
    """Same resolution bin/main uses."""
    return os.path.join("datasets", "graphs", "old", dataset + ".graph")


def read_gfa(path):
    """-> ({name: sequence length}, [(from, to)]), tolerating the '*' sequence placeholder."""
    lengths, links = {}, []

    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "S" and len(parts) > 2:
                lengths[parts[1]] = 0 if parts[2] == "*" else len(parts[2])
            elif parts[0] == "L" and len(parts) > 3:
                links.append((parts[1], parts[3]))

    return lengths, links


def longest_path_levels(lengths, links):
    """Kahn's algorithm carrying a longest path depth, the same pass main.c runs."""
    out = collections.defaultdict(list)
    indeg = collections.Counter()

    for u, v in links:
        out[u].append(v)
        indeg[v] += 1

    depth = {n: 0 for n in lengths}
    left = {n: indeg[n] for n in lengths}
    queue = collections.deque(n for n in lengths if left[n] == 0)
    seen = 0

    while queue:
        u = queue.popleft()
        seen += 1
        for v in out[u]:
            if depth[u] + 1 > depth[v]:
                depth[v] = depth[u] + 1
            left[v] -= 1
            if left[v] == 0:
                queue.append(v)

    if seen != len(lengths):
        raise ValueError("cycle detected, the graph is not a DAG")

    return depth


def levels_of(dataset):
    path = graph_path(dataset)
    if not os.path.exists(path):
        print(f"  {dataset}: {path} not found, skipped", file=sys.stderr)
        return None

    lengths, links = read_gfa(path)
    depth = longest_path_levels(lengths, links)

    num_levels = max(depth.values()) + 1
    width = collections.Counter()
    bases = collections.Counter()
    longest = collections.Counter()

    for node, d in depth.items():
        width[d] += 1
        bases[d] += lengths[node]
        longest[d] = max(longest[d], lengths[node])

    return [(d, width[d], bases[d], longest[d]) for d in range(num_levels)]


def summarise(dataset, rows, min_nodes):
    """The four numbers the report quotes in prose, printed so a run doubles as a sanity check."""
    nodes = sum(w for _, w, _, _ in rows)
    total_bases = sum(b for _, _, b, _ in rows)

    wide = [r for r in rows if r[1] >= min_nodes]
    thin = [r for r in rows if r[1] < min_nodes]

    print(f"  {dataset}: {nodes} nodes, {total_bases} bases, {len(rows)} levels, "
          f"max width {max(w for _, w, _, _ in rows)}")
    print(f"      width >= {min_nodes}: {len(wide):>5} levels, {sum(r[1] for r in wide):>6} nodes "
          f"({100.0 * sum(r[2] for r in wide) / max(total_bases, 1):.1f}% of the work)")
    print(f"      width <  {min_nodes}: {len(thin):>5} levels, {sum(r[1] for r in thin):>6} nodes "
          f"({100.0 * sum(r[2] for r in thin) / max(total_bases, 1):.1f}% of the work)")

    # The tail's critical path: even with a thread per node, a level still costs its longest node,
    # so this is the ceiling on parallelising the part the CPU takes.
    thin_work = sum(r[2] for r in thin)
    thin_path = sum(r[3] for r in thin)
    if thin_path:
        print(f"      tail critical path {100.0 * thin_path / max(thin_work, 1):.0f}% of tail work "
              f"-> at best {thin_work / thin_path:.2f}x from more threads")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("-o", "--outdir", default="outputs/report")
    ap.add_argument("--min-nodes", type=int, default=16,
                    help="HYBRID_MIN_NODES, only used for the printed summary (default: 16)")
    args = ap.parse_args()

    if not os.path.isdir("datasets"):
        sys.exit("run me from the repository root")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "graph_levels.csv")

    rows = []
    for dataset in args.datasets:
        levels = levels_of(dataset)
        if levels is None:
            continue
        summarise(dataset, levels, args.min_nodes)
        rows += [{"dataset": dataset, "level": d, "width": w, "bases": b, "maxlen": n}
                 for d, w, b, n in levels]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "level", "width", "bases", "maxlen"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
