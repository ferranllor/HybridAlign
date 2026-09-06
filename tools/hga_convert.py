#!/usr/bin/env python3
"""
Converts one of this project's datasets into the input format HGA expects, so the two aligners can
be run on exactly the same graph and reads.

    python3 tools/hga_convert.py 150_10 -n 256 -o outputs/hga

HGA's graph is base level: one vertex per nucleotide, held in CSR. Ours is segment level: one node
per sequence of up to 512 bases. The conversion expands every node of length L into a chain of L
vertices, and turns each of our edges into a single edge from the last vertex of the source to the
first vertex of the target. So the vertex count of the converted graph is the total base count of
ours, and the two systems end up computing DP matrices of exactly the same size -- which is what
makes their throughput figures comparable rather than merely similar.

Vertices are numbered in topological order, and a node's own chain is numbered consecutively.
HGA's kernel reads dp[inv[k]] for the *current* row, so an in-neighbour has to carry a lower id
than the vertex that reads it; a numbering that is not topological silently produces wrong scores
rather than an error.

Graph file (whitespace separated, exactly the order Align::input_graph reads it):

    num_v num_e
    inv[num_e]        in-neighbour ids, grouped by vertex
    inoff[num_v + 1]  CSR offsets into inv
    outv[num_e]       out-neighbour ids
    outoff[num_v + 1] CSR offsets into outv
    ref_graph[num_v]  one base per vertex

Read file: two whitespace separated tokens per read, a name and the sequence.

Run from the repository root.
"""

import argparse
import collections
import os
import sys

BASES = "ACGT"


def read_dataset(dataset):
    """-> ([(node_id, sequence)], [(from, to)], query string), in the file's own node order."""
    gpath = os.path.join("datasets", "graphs", "old", dataset + ".graph")
    spath = os.path.join("datasets", "sequences", "old", "S_" + dataset + ".seq")

    seqs, links = {}, []
    with open(gpath) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "S" and len(parts) > 2:
                seqs[parts[1]] = "" if parts[2] == "*" else parts[2]
            elif parts[0] == "L" and len(parts) > 3:
                links.append((parts[1], parts[3]))

    with open(spath) as f:
        fields = f.readline().split()
    if len(fields) < 3:
        sys.exit(f"cannot read a query out of {spath}")

    return seqs, links, fields[2]


def topological_order(seqs, links):
    """Kahn, same longest-path-free ordering the aligner itself relies on."""
    out = collections.defaultdict(list)
    indeg = collections.Counter()
    for u, v in links:
        out[u].append(v)
        indeg[v] += 1

    left = {n: indeg[n] for n in seqs}
    queue = collections.deque(sorted(n for n in seqs if left[n] == 0))
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in out[u]:
            left[v] -= 1
            if left[v] == 0:
                queue.append(v)

    if len(order) != len(seqs):
        sys.exit("cycle detected: HGA needs a DAG")

    return order


def build(seqs, links, order):
    """-> (bases, in_lists) with vertices numbered in topological order.

    A node of length L becomes vertices [base, base + L); vertex base + i has the single
    in-neighbour base + i - 1, except the first, which inherits the node's predecessors."""
    first, last, base = {}, {}, 0
    for name in order:
        length = len(seqs[name])
        if length == 0:
            continue                       # a '*' segment carries no cells, so it carries no vertex
        first[name] = base
        last[name] = base + length - 1
        base += length

    num_v = base
    bases = bytearray(num_v)
    in_lists = [[] for _ in range(num_v)]

    for name in order:
        if name not in first:
            continue
        start = first[name]
        seq = seqs[name]
        for i, ch in enumerate(seq):
            bases[start + i] = ord(ch if ch in BASES else "N")
            if i:
                in_lists[start + i].append(start + i - 1)

    dropped = 0
    for u, v in links:
        if u not in last or v not in first:
            dropped += 1
            continue
        in_lists[first[v]].append(last[u])

    if dropped:
        print(f"  {dropped} edge(s) touched an empty segment and were dropped", file=sys.stderr)

    return bases, in_lists


def write_graph(path, bases, in_lists):
    num_v = len(bases)

    # HGA wants both directions in CSR. The out lists are the transpose of the in lists.
    out_lists = [[] for _ in range(num_v)]
    for v, ins in enumerate(in_lists):
        for u in ins:
            out_lists[u].append(v)

    num_e = sum(len(x) for x in in_lists)

    def csr(lists):
        flat, off = [], [0]
        for l in lists:
            flat.extend(l)
            off.append(len(flat))
        return flat, off

    inv, inoff = csr(in_lists)
    outv, outoff = csr(out_lists)

    with open(path, "w") as f:
        f.write(f"{num_v} {num_e}\n")
        for arr in (inv, inoff, outv, outoff):
            f.write("\n".join(map(str, arr)))
            f.write("\n")
        f.write("\n".join(chr(b) for b in bases))
        f.write("\n")

    return num_v, num_e


def write_reads(path, query, num_reads):
    """The same synthetic batch multi_build_reads() makes on our side: read 0 is the query, and
    every later read is the query with M/20 substitutions from the same FNV walk, so both aligners
    see identical input."""
    m = len(query)
    with open(path, "w") as f:
        for r in range(num_reads):
            dst = list(query)
            if r > 0:
                h = (2166136261 ^ r) & 0xFFFFFFFF
                for _ in range(m // 20):
                    h = (h * 16777619 + 2654435761) & 0xFFFFFFFF
                    dst[h % m] = BASES[(h >> 16) & 3]
            f.write(f"read{r}\t{''.join(dst)}\n")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("-n", "--num-reads", type=int, default=1)
    ap.add_argument("-o", "--outdir", default="outputs/hga")
    args = ap.parse_args()

    if not os.path.isdir("datasets"):
        sys.exit("run me from the repository root")
    os.makedirs(args.outdir, exist_ok=True)

    seqs, links, query = read_dataset(args.dataset)
    order = topological_order(seqs, links)
    bases, in_lists = build(seqs, links, order)

    gpath = os.path.join(args.outdir, f"{args.dataset}.hga.graph")
    rpath = os.path.join(args.outdir, f"{args.dataset}.r{args.num_reads}.hga.reads")

    num_v, num_e = write_graph(gpath, bases, in_lists)
    m = write_reads(rpath, query, args.num_reads)

    print(f"  {args.dataset}: {len(seqs)} segments -> {num_v} vertices, {num_e} edges")
    print(f"  {args.num_reads} read(s) of {m} bases")
    print(f"  cells per read: {num_v * m / 1e9:.3f} G")
    print(f"  {gpath}")
    print(f"  {rpath}")


if __name__ == "__main__":
    main()
