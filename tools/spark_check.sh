#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# One command to run on the DGX Spark. It collects everything needed to explain why the managed
# memory hybrid has unstable run times there, and whether the two candidate fixes remove it:
#
#   1. tools/bin/spark_probe          - what the machine's memory actually does (bandwidths, the
#                                       cost and stability of a GPU->CPU hand over per memory kind,
#                                       per level overheads)
#   2. bin/main <ds> 2 1|2|3          - the three hybrid memory variants, repeated, with
#                                       HYBRID_STATS=1 so each run reports launch / wait / cpu
#   3. bin/main <ds> 1 6 and 0 3      - the two versions that are already stable, as a reference
#
#   tools/spark_check.sh                       # 150_10, 5 repeats
#   tools/spark_check.sh -d 500_10 -n 10
#   tools/spark_check.sh -o outputs/spark      # where the log goes
#
# Everything lands in one file; send that file back.
# ------------------------------------------------------------------------------------------------
set -u
export LC_ALL=C

DATASET="150_10"
REPEATS=5
OUT="outputs/spark"

while getopts "d:n:o:h" opt; do
    case $opt in
        d) DATASET="$OPTARG" ;;
        n) REPEATS="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) sed -n '2,18p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

[ -d datasets ] || { echo "run me from the repository root"; exit 2; }
mkdir -p "$OUT"
LOG="$OUT/spark_check_${DATASET}.log"

{
    echo "==================================================================="
    echo " spark_check   dataset=$DATASET  repeats=$REPEATS   $(date -Is)"
    echo " host: $(uname -srm)   $(nproc) cores"
    echo "==================================================================="
} > "$LOG"

echo "building..."
make >/dev/null 2>&1 || { echo "make failed"; exit 2; }
make -C tools spark_probe >/dev/null 2>&1 || { echo "probe build failed"; exit 2; }

echo "1/3 machine probe"
{
    echo
    echo "################ machine probe ################"
    ./tools/bin/spark_probe 2>&1
} >> "$LOG"

run_version() {
    local mode="$1" version="$2" label="$3"
    echo "  $label"
    {
        echo
        echo "################ $label  (mode $mode version $version) ################"
        for i in $(seq 1 "$REPEATS"); do
            HYBRID_STATS=1 ./bin/main "$DATASET" "$mode" "$version" 2>&1 \
                | grep -E "^\[hybrid\]|^Iteration|^Arithmetic|Identity Score"
        done
    } >> "$LOG"
}

echo "2/3 hybrid memory variants"
run_version 2 1 "hybrid_unified  (managed memory, one sync per CPU level)"
run_version 2 2 "hybrid_pinned   (pinned host memory, pages never move)"
run_version 2 3 "hybrid_advised  (managed + preferred location host, accessed by device)"

echo "3/3 stable references"
run_version 1 6 "gpu shared_mem  (explicit copies, was stable)"
run_version 0 3 "cpu simd_parallel_node (no CUDA, was stable)"

# --------------------------------------------------------------------------- summary
{
    echo
    echo "################ summary: mean and spread of Arithmetic Mean ################"
    awk '
        /^################ / { name = $0; sub(/^################ /, "", name); sub(/ ################$/, "", name); next }
        /^Arithmetic Mean:/ {
            v = $3 + 0
            n[name]++; s[name] += v; q[name] += v * v
            if (n[name] == 1 || v < mn[name]) mn[name] = v
            if (n[name] == 1 || v > mx[name]) mx[name] = v
        }
        END {
            printf "%-62s %8s %8s %8s %8s %7s\n", "version", "runs", "mean", "min", "max", "spread"
            for (k in n) {
                m = s[k] / n[k]
                sd = sqrt(q[k] / n[k] - m * m)
                printf "%-62s %8d %8.4f %8.4f %8.4f %6.1f%%\n", k, n[k], m, mn[k], mx[k], 100 * sd / m
            }
        }' "$LOG"
} >> "$LOG"

echo
tail -n 12 "$LOG"
echo
echo "full log -> $LOG   (send this file back)"
