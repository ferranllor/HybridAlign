#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Timing sweep for the no-copy versions (mode 3) across the three SHARED_MEM_KIND allocators
# (advised, pinned, managed). This is the one axis tools/benchmark.py does not already sweep,
# since SHARED_MEM_KIND is read from the environment, not passed as a CLI version number.
# Meant to be run once per machine (Baldo, DGX Spark) for the evaluation's No-copy table.
#
#   tools/bench_nocopy.sh                              # default dataset 150_10, versions 0 2 6-10
#   tools/bench_nocopy.sh -d 500_10 -v "0 6"
#   tools/bench_nocopy.sh -n 5 -o outputs/nocopy
#
# Writes one CSV per kind (outputs/nocopy/nocopy_<kind>_<dataset>.csv, via tools/benchmark.py)
# plus a plain-text summary table to stdout and outputs/nocopy/summary_<dataset>.log.
# Must be run from the repository root.
# ------------------------------------------------------------------------------------------------
set -u
export LC_ALL=C

DATASET="150_10"
VERSIONS="0 2 6 7 8 9 10"
REPEATS=3
KINDS="advised pinned managed"
OUT="outputs/nocopy"

while getopts "d:v:n:k:o:h" opt; do
    case $opt in
        d) DATASET="$OPTARG" ;;
        v) VERSIONS="$OPTARG" ;;
        n) REPEATS="$OPTARG" ;;
        k) KINDS="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) sed -n '2,15p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

[ -d datasets ] || { echo "run me from the repository root"; exit 2; }
[ -x ./bin/main ] || { echo "./bin/main not found - run make first"; exit 2; }
mkdir -p "$OUT"

LOG="$OUT/summary_${DATASET}.log"
{
    echo "==================================================================="
    echo " bench_nocopy  dataset=$DATASET  versions=$VERSIONS  repeats=$REPEATS  $(date -Is)"
    echo " host: $(uname -a)"
    echo "==================================================================="
} > "$LOG"

for kind in $KINDS; do
    echo "== SHARED_MEM_KIND=$kind ==" | tee -a "$LOG"
    csv="$OUT/nocopy_${kind}_${DATASET}.csv"

    SHARED_MEM_KIND="$kind" python3 tools/benchmark.py \
        -d "$DATASET" --cpu --gpu --hybrid --nocopy $VERSIONS -o "$csv" \
        -t 1800 | tee -a "$LOG"

    # mean over timed iterations (iter >= 0), one line per version, appended to the log
    awk -F, -v kind="$kind" '
        NR == 1 { next }
        $2 == 3 && $5 + 0 >= 0 { sum[$3] += $6; n[$3]++; label[$3] = $4 }
        END {
            for (v in sum)
                printf "  SHARED_MEM_KIND=%-8s v%-2s (%-14s) mean %8.4f s  (n=%d)\n", \
                       kind, v, label[v], sum[v] / n[v], n[v]
        }
    ' "$csv" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "summary -> $LOG"
