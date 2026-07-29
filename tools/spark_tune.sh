#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Next iteration after spark_check: with the memory question settled, the CPU levels are the bigger
# half of a hybrid alignment and the only remaining source of run to run spread. Two knobs decide
# how much of the graph they get and how well they run:
#
#   1. HYBRID_MIN_NODES - the level size at which work switches from the GPU to the CPU. The right
#      value is a property of the machine (how fast its CPU is relative to its GPU), so it has to be
#      swept where it will run, not where it was written.
#   2. thread placement  - on a big.LITTLE part, OpenMP threads landing on the small cluster make
#      the CPU phase both slower and erratic. Tried through OMP_PLACES, no rebuild needed.
#
#   tools/spark_tune.sh                        # sweep both, hybrid version 3, 150_10
#   tools/spark_tune.sh -v 2 -d 500_10 -n 3
#   tools/spark_tune.sh -t "1 2 4 8 16 32"     # thresholds to try
#
# Restores the default build at the end. Results in outputs/spark/spark_tune_<dataset>.log.
# ------------------------------------------------------------------------------------------------
set -u
export LC_ALL=C

DATASET="150_10"
VERSION=3
REPEATS=3
THRESHOLDS="1 2 4 8 16 32 64"
OUT="outputs/spark"

while getopts "d:v:n:t:o:h" opt; do
    case $opt in
        d) DATASET="$OPTARG" ;;
        v) VERSION="$OPTARG" ;;
        n) REPEATS="$OPTARG" ;;
        t) THRESHOLDS="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) sed -n '2,18p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

[ -d datasets ] || { echo "run me from the repository root"; exit 2; }
mkdir -p "$OUT"
LOG="$OUT/spark_tune_${DATASET}.log"

{
    echo "==================================================================="
    echo " spark_tune  dataset=$DATASET  hybrid version=$VERSION  repeats=$REPEATS  $(date -Is)"
    echo "==================================================================="
    echo
    echo "--- cores ---"
    lscpu -e 2>/dev/null | head -25 || true
} > "$LOG"

# mean of the "Arithmetic Mean" lines of N runs, plus the mean cpu/wait split from HYBRID_STATS
measure() {
    local tag="$1"
    local means=""
    for i in $(seq 1 "$REPEATS"); do
        means="$means $(HYBRID_STATS=1 ./bin/main "$DATASET" 2 "$VERSION" 2>>"$LOG" \
                        | awk '/^Arithmetic Mean:/ { print $3 }')"
    done
    # the label is passed as a variable: it contains spaces, awk must not read it as data
    printf "%s\n" "$means" | awk -v tag="$tag" '{
        s = 0; n = 0; mn = 0; mx = 0;
        for (i = 1; i <= NF; i++) {
            v = $i + 0; s += v; n++;
            if (n == 1 || v < mn) mn = v;
            if (n == 1 || v > mx) mx = v;
        }
        if (n) printf "  %-34s mean %8.4f s   min %8.4f   max %8.4f\n", tag, s/n, mn, mx
    }' | tee -a "$LOG"
}

echo "== sweeping HYBRID_MIN_NODES ==" | tee -a "$LOG"
for t in $THRESHOLDS; do
    rm -f obj/hybrid_pinned.o bin/main
    make EXTRA="-DHYBRID_MIN_NODES=$t" >/dev/null 2>&1 || { echo "build failed for $t"; continue; }
    measure "HYBRID_MIN_NODES=$t"
done

echo | tee -a "$LOG"
echo "== thread placement (rebuilt with the default threshold) ==" | tee -a "$LOG"
rm -f obj/hybrid_pinned.o bin/main
make >/dev/null 2>&1

NPROC=$(nproc)
HALF=$((NPROC / 2))

unset OMP_PLACES OMP_PROC_BIND
measure "default placement"

export OMP_PROC_BIND=close OMP_PLACES=cores
measure "bind=close places=cores"

export OMP_PROC_BIND=close OMP_PLACES="{0}:$HALF"
measure "first half (cores 0-$((HALF - 1)))"

export OMP_PROC_BIND=close OMP_PLACES="{$HALF}:$HALF"
measure "second half (cores $HALF-$((NPROC - 1)))"

unset OMP_PLACES OMP_PROC_BIND

echo | tee -a "$LOG"
echo "log -> $LOG"
