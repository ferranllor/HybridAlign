#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Timeline profile of one or more versions with Nsight Systems, reduced to the three numbers the
# report iterates on: how long the kernels run, how long the copies run, and how much of the wall
# time is neither (launch gaps, syncs, CPU work).
#
#   tools/profile.sh                                  # default datasets/versions
#   tools/profile.sh -d 150_10 -v "1:8 2:0 2:1"       # versions are mode:version, like verify.sh
#   tools/profile.sh -o outputs/profile               # where reports and CSVs go
#
# Writes, per run:  <out>/<dataset>_<mode>.<version>.nsys-rep   (open in the Nsight Systems GUI)
#                   <out>/<dataset>_<mode>.<version>_*.csv      (per report tables)
# and a combined <out>/summary.csv.
#
# NOTE: bin/main aligns the sequence several times per process (one verification run, WARMUP warmup
# iterations and NITER timed ones), so the nsys totals cover all of them. The *_per_align columns
# divide by the number of alignments actually observed in the log.
#
# Must be run from the repository root, with nothing else using the GPU.
# ------------------------------------------------------------------------------------------------
set -u

# awk/printf must use dots for decimals: this writes a comma separated file, and a comma
# decimal separator would corrupt it (and break the parsing of main.c's timings).
export LC_ALL=C

DATASETS="150_10_small 150_10"
VERSIONS="1:0 1:5 1:7 1:8 1:9 1:10 2:0 2:1"
OUT="outputs/profile"

while getopts "d:v:o:h" opt; do
    case $opt in
        d) DATASETS="$OPTARG" ;;
        v) VERSIONS="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        h) sed -n '2,16p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

[ -d datasets ] || { echo "run me from the repository root"; exit 2; }
command -v nsys >/dev/null || { echo "nsys not found"; exit 2; }
mkdir -p "$OUT"

SUMMARY="$OUT/summary.csv"
HEADER="dataset,mode,version,wall_s,alignments,kernel_ms_per_align,memcpy_ms_per_align,kernel_launches_per_align,memcpy_calls_per_align,kernel_ms_total,memcpy_ms_total"

# Rows accumulate across invocations so several datasets can be profiled in separate runs.
# Delete the file to start a fresh table.
[ -f "$SUMMARY" ] || echo "$HEADER" > "$SUMMARY"

# sums column 2 (Total Time) of an nsys csv report, which is in nanoseconds
sum_ns() {
    [ -f "$1" ] || { echo 0; return; }
    awk -F',' 'NR>1 && $2 ~ /^[0-9]/ { s += $2 } END { printf "%.0f", s+0 }' "$1"
}
sum_calls() {
    [ -f "$1" ] || { echo 0; return; }
    awk -F',' 'NR>1 && $3 ~ /^[0-9]/ { s += $3 } END { printf "%.0f", s+0 }' "$1"
}

for ds in $DATASETS; do
    for mv in $VERSIONS; do
        case "$mv" in
            *:*) m="${mv%%:*}"; v="${mv##*:}" ;;
            *)   m=1;           v="$mv" ;;
        esac

        base="$OUT/${ds}_${m}.${v}"
        echo "== profiling $ds mode=$m version=$v"

        nsys profile --force-overwrite true -o "$base" --trace=cuda \
             ./bin/main "$ds" "$m" "$v" > "$base.log" 2>&1

        if [ ! -f "$base.nsys-rep" ]; then
            echo "   nsys failed, see $base.log"
            continue
        fi

        nsys stats --force-export true --format csv --output "$base" \
             --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
             "$base.nsys-rep" > /dev/null 2>&1

        kern_ns=$(sum_ns "${base}_cuda_gpu_kern_sum.csv")
        mem_ns=$(sum_ns "${base}_cuda_gpu_mem_time_sum.csv")
        kern_calls=$(sum_calls "${base}_cuda_gpu_kern_sum.csv")
        mem_calls=$(sum_calls "${base}_cuda_gpu_mem_time_sum.csv")

        # mean of the timed iterations that main.c prints
        wall=$(awk '/Iteration/ && $2 >= 0 { s += $4; n++ } END { if (n) printf "%.6f", s/n; else print 0 }' \
               FS='[ s]+' "$base.log")

        # one alignment per "Iteration" line, plus the verification run at the start
        aligns=$(( $(grep -c "^Iteration" "$base.log") + 1 ))
        [ "$aligns" -lt 1 ] && aligns=1

        read kms mms kl mc <<EOF
$(awk -v k="$kern_ns" -v m="$mem_ns" -v kc="$kern_calls" -v mc="$mem_calls" -v a="$aligns" \
      'BEGIN { printf "%.3f %.3f %.1f %.1f", k/1e6/a, m/1e6/a, kc/a, mc/a }')
EOF

        echo "$ds,$m,$v,$wall,$aligns,$kms,$mms,$kl,$mc,$(awk -v k="$kern_ns" 'BEGIN{printf "%.3f", k/1e6}'),$(awk -v m="$mem_ns" 'BEGIN{printf "%.3f", m/1e6}')" >> "$SUMMARY"
        printf "   wall %ss/align   kernels %s ms (%s launches)   copies %s ms (%s calls)\n" \
               "$wall" "$kms" "$kl" "$mms" "$mc"
    done
done

echo
echo "summary -> $SUMMARY"
