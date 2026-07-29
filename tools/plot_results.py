#!/usr/bin/env python3
"""
Plots the CSV produced by tools/benchmark.py.

    python3 tools/benchmark.py                 # -> outputs/timings.csv
    python3 tools/plot_results.py              # -> outputs/fig_runtime.(png|pdf)
                                               #    outputs/fig_speedup.(png|pdf)
    python3 tools/plot_results.py --dark       # dark-surface variant
    python3 tools/plot_results.py -i outputs/timings.csv -o outputs

Two figures, one panel per dataset:
  * fig_runtime  - mean wall time per implementation (warmup iteration dropped, whiskers = min/max)
  * fig_speedup  - the same runs as speedup over the CPU sequential baseline

Needs matplotlib:  pip install matplotlib     (or: python3 -m venv .venv && .venv/bin/pip install matplotlib)
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

# Palette: slots 1-3 of the validated categorical set (blue, orange, aqua), used verbatim - the
# three slots that clear the all-pairs gates in both modes. Identity is carried by a legend plus the
# fixed CPU / GPU / hybrid grouping, never by colour alone.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dedcd6",
                  cpu="#eb6834", gpu="#2a78d6", hybrid="#1baf7a", nocopy="#eda100"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37",
                  cpu="#d95926", gpu="#3987e5", hybrid="#199e70", nocopy="#c98500"),
}

MODE_NAMES = {0: "CPU", 1: "GPU", 2: "HYB", 3: "NCP"}
MODE_COLORS = {0: "cpu", 1: "gpu", 2: "hybrid", 3: "nocopy"}


def load(path):
    """rows -> {(dataset, mode, version, label): [times]} keeping the file's ordering."""
    runs = defaultdict(list)
    order = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["iter"]) < 0:          # warmup
                continue
            key = (r["dataset"], int(r["mode"]), int(r["version"]), r["label"])
            if key not in runs:
                order.append(key)
            runs[key].append(float(r["time_s"]))
    if not runs:
        sys.exit("no timed iterations found in the CSV")
    return runs, order


def implementations(order):
    """Fixed display order: CPU versions first, then GPU versions - never re-sorted by value,
    so a given implementation keeps its row (and its colour) in every panel."""
    impls = sorted({(m, v, lab) for (_, m, v, lab) in order})
    return impls


def fmt_time(v):
    if v >= 1:
        return f"{v:.3g} s"
    if v >= 1e-3:
        return f"{v * 1e3:.3g} ms"
    return f"{v * 1e6:.3g} us"


def panel(ax, labels, values, lo, hi, colors, theme, xlabel, logx, ref=None, fmt=fmt_time):
    y = list(range(len(labels)))

    # A bar encodes magnitude by *length*, which is only meaningful on a linear axis anchored at
    # zero. When the runtimes span orders of magnitude the axis has to be logarithmic, and then
    # the honest mark is a dot: position, not length, carries the value.
    if logx:
        for i, (v, c) in enumerate(zip(values, colors)):
            ax.plot([v], [i], "o", color=c, markersize=9, zorder=3,
                    markeredgecolor=theme["surface"], markeredgewidth=1.5)
    else:
        ax.barh(y, values, height=0.62, color=colors, zorder=3)

    # min/max whisker, drawn in secondary ink so it reads as annotation, not as a series
    for i, (v, a, b) in enumerate(zip(values, lo, hi)):
        if b > a:
            ax.plot([a, b], [i, i], color=theme["ink2"], lw=1.2, zorder=2,
                    solid_capstyle="butt", alpha=0.85)

    if ref is not None:
        ax.axvline(ref, color=theme["ink2"], lw=1, ls=(0, (4, 3)), zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=theme["ink2"], fontsize=8)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=theme["ink2"], fontsize=8)
    ax.tick_params(axis="x", colors=theme["ink2"], labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=theme["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])

    # direct labels: text stays in ink tokens, the mark carries the identity colour
    span = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(v * 1.18 if logx else v + span * 0.02, i, fmt(v),
                va="center", ha="left", fontsize=7.5, color=theme["ink"])
    if logx:
        ax.set_xlim(min(lo) / 2.0, span * 4.0)
    else:
        ax.set_xlim(0, span * 1.24)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default="outputs/timings.csv")
    ap.add_argument("-o", "--outdir", default="outputs")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--baseline", default="sequential",
                    help="label used as the speedup baseline (default: CPU sequential)")
    args = ap.parse_args()

    theme = THEME["dark" if args.dark else "light"]
    runs, order = load(args.input)
    datasets = list(dict.fromkeys(ds for (ds, _, _, _) in order))
    impls = implementations(order)
    os.makedirs(args.outdir, exist_ok=True)

    suffix = "_dark" if args.dark else ""

    for kind in ("runtime", "speedup"):
        n = len(datasets)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 0.34 * len(impls) + 2.1), squeeze=False)
        fig.patch.set_facecolor(theme["surface"])

        for ax, ds in zip(axes[0], datasets):
            ax.set_facecolor(theme["surface"])
            labels, values, lo, hi, colors = [], [], [], [], []

            base = None
            for (m, v, lab) in impls:
                ts = runs.get((ds, m, v, lab))
                if m == 0 and lab == args.baseline and ts:
                    base = sum(ts) / len(ts)

            for (m, v, lab) in impls:
                ts = runs.get((ds, m, v, lab))
                if not ts:
                    continue
                mean = sum(ts) / len(ts)
                labels.append(f"{MODE_NAMES.get(m, m)} {v} {lab}")
                colors.append(theme[MODE_COLORS.get(m, "gpu")])
                if kind == "runtime":
                    values.append(mean)
                    lo.append(min(ts))
                    hi.append(max(ts))
                else:
                    if not base:
                        continue
                    values.append(base / mean)
                    lo.append(base / max(ts))
                    hi.append(base / min(ts))

            if not values:
                ax.set_visible(False)
                continue

            if kind == "runtime":
                logx = max(values) / max(min(values), 1e-12) > 20
                panel(ax, labels, values, lo, hi, colors, theme,
                      "mean wall time (s, log)" if logx else "mean wall time (s)", logx)
            else:
                panel(ax, labels, values, lo, hi, colors, theme,
                      f"speedup over CPU {args.baseline}", False, ref=1.0,
                      fmt=lambda v: f"{v:.2f}x")

            ax.set_title(ds, color=theme["ink"], fontsize=10, loc="left", pad=8)

        modes_present = sorted({m for (m, _, _) in impls})
        handles = [plt.Rectangle((0, 0), 1, 1, color=theme[MODE_COLORS.get(m, "gpu")])
                   for m in modes_present]
        leg = fig.legend(handles, [MODE_NAMES.get(m, str(m)) for m in modes_present],
                         loc="upper right", frameon=False,
                         ncol=len(modes_present), fontsize=8, bbox_to_anchor=(0.995, 0.995))
        for text in leg.get_texts():
            text.set_color(theme["ink2"])

        title = ("Alignment wall time per implementation"
                 if kind == "runtime" else
                 f"Speedup over CPU {args.baseline}")
        fig.suptitle(title, color=theme["ink"], fontsize=12, x=0.006, ha="left", y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        for ext in ("png", "pdf"):
            path = os.path.join(args.outdir, f"fig_{kind}{suffix}.{ext}")
            fig.savefig(path, dpi=200, facecolor=theme["surface"])
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
