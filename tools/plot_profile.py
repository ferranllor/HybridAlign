#!/usr/bin/env python3
"""
Plots the summary produced by tools/profile.sh: for every version, the measured wall time next to
the two things that fill it - time spent in kernels and time spent moving memory.

    tools/profile.sh                           # -> outputs/profile/summary.csv
    python3 tools/plot_profile.py              # -> outputs/fig_profile.(png|pdf)
    python3 tools/plot_profile.py --dark

Kernels and copies run on different streams, so they overlap: the three bars are three measured
quantities on the same scale, NOT a breakdown that adds up to the wall time. That is the point of
the figure - a version whose kernel and copy bars are both far below its wall bar is spending its
time on neither (launch gaps, syncs, CPU work).

Needs matplotlib:  pip install matplotlib
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is not installed: pip install matplotlib")

# Slots 1-3 of the validated categorical palette (blue, orange, aqua).
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd6",
                  wall="#2a78d6", kernel="#eb6834", memcpy="#1baf7a"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37",
                  wall="#3987e5", kernel="#d95926", memcpy="#199e70"),
}

MODE_NAMES = {0: "CPU", 1: "GPU", 2: "HYB"}
SERIES = [("wall", "wall time"), ("kernel", "kernels"), ("memcpy", "memory transfers")]


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "dataset": r["dataset"],
                    "mode": int(r["mode"]),
                    "version": int(r["version"]),
                    "wall": float(r["wall_s"]) * 1000.0,
                    "kernel": float(r["kernel_ms_per_align"]),
                    "memcpy": float(r["memcpy_ms_per_align"]),
                    "launches": float(r["kernel_launches_per_align"]),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        sys.exit("no usable rows in the profile summary")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default="outputs/profile/summary.csv")
    ap.add_argument("-o", "--outdir", default="outputs")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    theme = THEME["dark" if args.dark else "light"]
    rows = load(args.input)
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    os.makedirs(args.outdir, exist_ok=True)

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    n = len(datasets)
    height = max(len(v) for v in by_ds.values()) * 0.75 + 2.6
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, height), squeeze=False)
    fig.patch.set_facecolor(theme["surface"])

    for ax, ds in zip(axes[0], datasets):
        ax.set_facecolor(theme["surface"])
        items = sorted(by_ds[ds], key=lambda r: (r["mode"], r["version"]))
        labels = [f"{MODE_NAMES.get(r['mode'], r['mode'])} {r['mode']}.{r['version']}\n"
                  f"{r['launches']:.0f} launches" for r in items]

        bar_h = 0.26
        for si, (key, _) in enumerate(SERIES):
            offs = (si - 1) * (bar_h + 0.02)
            ax.barh([i + offs for i in range(len(items))],
                    [r[key] for r in items], height=bar_h, color=theme[key], zorder=3)
            for i, r in enumerate(items):
                ax.text(r[key] + max(x["wall"] for x in items) * 0.012, i + offs,
                        f"{r[key]:.0f}", va="center", ha="left", fontsize=7, color=theme["ink"])

        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(labels, color=theme["ink2"], fontsize=8)
        ax.set_ylim(len(items) - 0.5, -0.5)
        ax.set_xlabel("ms per alignment", color=theme["ink2"], fontsize=8)
        ax.tick_params(axis="x", colors=theme["ink2"], labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=theme["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])
        ax.set_xlim(0, max(r["wall"] for r in items) * 1.2)
        ax.set_title(ds, color=theme["ink"], fontsize=10, loc="left", pad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=theme[k]) for k, _ in SERIES]
    # below the panels: with one narrow panel a top-right legend runs into the title
    leg = fig.legend(handles, [name for _, name in SERIES], loc="lower center", frameon=False,
                     ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(theme["ink2"])

    fig.suptitle("Where each version spends its time (kernels and copies overlap)",
                 color=theme["ink"], fontsize=12, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    suffix = "_dark" if args.dark else ""
    for ext in ("png", "pdf"):
        path = os.path.join(args.outdir, f"fig_profile{suffix}.{ext}")
        fig.savefig(path, dpi=200, facecolor=theme["surface"])
        print("wrote", path)


if __name__ == "__main__":
    main()
