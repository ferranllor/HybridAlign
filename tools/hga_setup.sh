#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Fetches and builds HGA, the state-of-the-art heterogeneous sequence-to-graph aligner this project
# is compared against (Feng and Luo, ICPP 2021).
#
#   tools/hga_setup.sh                 # clone if needed, build for this machine's GPU
#   tools/hga_setup.sh -a sm_80        # force an architecture (A30 = sm_80, GB10 = sm_121)
#   tools/hga_setup.sh -d outputs/hga  # where the working copy lives
#
# The upstream Makefile pins -arch=compute_75 -code=sm_75, which is a Turing card and will not run
# on anything we test on, so the arch is overridden here rather than by editing their tree.
#
# Leaves the binary at <dir>/hga-src/bin/hga, which is where tools/report_bench.py looks for it.
# A copy of the sources is kept under outputs/, which is gitignored, so their code is never
# committed into this repository.
#
# Run from the repository root.
# ------------------------------------------------------------------------------------------------
set -eu

REPO="https://github.com/RapidsAtHKUST/hga.git"
DIR="outputs/hga"
ARCH=""

while getopts "a:d:h" opt; do
    case $opt in
        a) ARCH="$OPTARG" ;;
        d) DIR="$OPTARG" ;;
        h) sed -n '2,17p' "$0"; exit 0 ;;
        *) exit 2 ;;
    esac
done

[ -d datasets ] || { echo "run me from the repository root"; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc not found"; exit 2; }

SRC="$DIR/hga-src"
mkdir -p "$DIR"

if [ ! -d "$SRC" ]; then
    echo "cloning HGA into $SRC"
    git clone --depth 1 "$REPO" "$SRC"
else
    echo "reusing the working copy in $SRC"
fi

# Guess the architecture from the card in front of us. compute capability 8.0 -> sm_80.
if [ -z "$ARCH" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)
    if [ -n "${CC:-}" ]; then
        ARCH="sm_${CC}"
    else
        echo "could not detect the GPU, pass -a sm_XX"; exit 2
    fi
fi

echo "building for $ARCH"
mkdir -p "$SRC/bin"
make -C "$SRC" clean >/dev/null 2>&1 || true
make -C "$SRC" all CUFLAGS="-arch=$ARCH -Xcompiler -fopenmp"

echo
echo "built $SRC/bin/hga"
echo "now run:  python3 tools/report_bench.py -m <machine>     (the hga part picks it up)"
