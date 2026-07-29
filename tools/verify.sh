#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Correctness sweep: every GPU version against the CPU sequential reference, on every dataset,
# optionally for several BLOCKSIZE values (that is what exercises the banding/striding of
# version 6 - a band is only split when BLOCKSIZE < min(M, N)).
#
#   tools/verify.sh                                   # default datasets, default BLOCKSIZE
#   tools/verify.sh -d "30_5 500_10" -v "1:5 1:6"     # pick datasets / versions
#   tools/verify.sh -v "2:0"                          # hybrid only (versions are mode:version)
#   tools/verify.sh -b "16 64 100 480"                # sweep BLOCKSIZE (rebuilds the harness)
#   tools/verify.sh -o outputs/verify.csv             # where the CSV goes
#
# Must be run from the repository root (dataset paths are relative).
# ------------------------------------------------------------------------------------------------
set -u

DATASETS="5_15 20_10 30_5 150_10_small 500_10"
VERSIONS="1:0 1:2 1:3 1:4 1:5 1:6 2:0 2:1 2:2 2:3"
BLOCKSIZES=""                       # empty -> whatever include/cuda_utils.cuh defines
OUT="outputs/verify.csv"

while getopts "d:v:b:o:h" opt; do
    case $opt in
        d) DATASETS="$OPTARG" ;;
        v) VERSIONS="$OPTARG" ;;
        b) BLOCKSIZES="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) sed -n '2,14p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

[ -d datasets ] || { echo "run me from the repository root"; exit 2; }
mkdir -p "$(dirname "$OUT")"

echo "dataset,mode,version,name,blocksize,nodes,M,bad_nodes,bad_cells,bad_max_scores,cpu_len,gpu_len,cpu_score,gpu_score,identical_alignment,ok" > "$OUT"

fails=0

run_sweep() {
    local bs_label="$1"
    for ds in $DATASETS; do
        for mv in $VERSIONS; do
            # versions are given as mode:version, matching bin/main (1 = GPU, 2 = hybrid)
            case "$mv" in
                *:*) m="${mv%%:*}"; v="${mv##*:}" ;;
                *)   m=1;           v="$mv" ;;
            esac

            printf "  %-16s %s.%-2s BLOCKSIZE=%-4s " "$ds" "$m" "$v" "$bs_label"
            line=$(./tools/bin/verify_dp "$ds" "$m" "$v" 0 --csv 2>/dev/null)
            rc=$?
            if [ -z "$line" ]; then
                echo "CRASH/ERROR"
                echo "$ds,$m,$v,,$bs_label,,,,,,,,,,,-1" >> "$OUT"
                fails=$((fails + 1))
                continue
            fi
            echo "$line" >> "$OUT"
            if [ $rc -eq 0 ]; then echo "PASS"; else echo "FAIL  ($line)"; fails=$((fails + 1)); fi
        done
    done
}

if [ -z "$BLOCKSIZES" ]; then
    echo "== building harness (default BLOCKSIZE) =="
    build_log=$(mktemp)
    make -C tools >"$build_log" 2>&1 || { cat "$build_log"; rm -f "$build_log"; exit 2; }
    rm -f "$build_log"
    run_sweep "default"
else
    for bs in $BLOCKSIZES; do
        echo "== building harness with BLOCKSIZE=$bs =="
        build_log=$(mktemp)
        make -C tools clean >/dev/null 2>&1
        make -C tools BLOCKSIZE="$bs" >"$build_log" 2>&1 || { cat "$build_log"; rm -f "$build_log"; exit 2; }
        rm -f "$build_log"
        run_sweep "$bs"
    done
    make -C tools clean >/dev/null 2>&1
fi

echo
echo "results -> $OUT"
if [ $fails -eq 0 ]; then echo "ALL PASS"; else echo "$fails FAILING CONFIGURATIONS"; fi
exit $((fails > 0))
