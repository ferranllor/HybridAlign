# HybridAlign

Sequence-to-graph alignment on CPU-GPU heterogeneous systems. One binary, `bin/main`, holds every implementation of the same alignment: CPU (sequential, SIMD, OpenMP), GPU (a ladder of CUDA versions from a naive one up to warp-per-node with register shuffles), and hybrid ones that split the graph between the two processors by level width.

The report and the slides that come out of this code are in [report/](report/).

## Requirements

* `gcc` with OpenMP
* the CUDA toolkit (`nvcc`) and an NVIDIA GPU
* for the plots and the sweep scripts: `python3` with `matplotlib`
* for `tools/profile.sh` only: Nsight Systems (`nsys`)

## Build

From the project root:

```sh
make                    # -> bin/main
make clean && make      # full rebuild
```

The makefile compiles for `-arch=sm_120`. If you are on another card, change line 9 in [makefile](makefile)

Compile-time knobs go through `EXTRA`, which is appended to the nvcc flags:

```sh
make EXTRA="-DHYBRID_MIN_NODES=32"   # level width at which the hybrid hands over to the CPU
make EXTRA="-DBLOCKSIZE=64"          # threads per node in the banded kernels
```

`WARMUP`, `NITER` (how many iterations each run times), `BATCHSIZE` and the rest of the tuning constants live in [include/definitions.h](include/definitions.h).

## Run

```sh
./bin/main <dataset> <mode> <version>
```

for example:

```sh
./bin/main 150_10 2 6      # hybrid, merged_req kernel on the dense levels, CPU on the tail
```

Every run does the same thing: it loads the graph and the read, does one verification alignment and prints both aligned strings, then runs `WARMUP` warmup iterations plus `NITER` timed ones and prints the per-iteration times and their arithmetic mean.

To see the modes and versions the binary was built with, pass an invalid pair, and a help messahe will be shown with the available versions:

```sh
./bin/main 150_10 -1 -1
```

### Datasets

`<dataset>` is the name of a file *pair*, not a path: `bin/main` reads `datasets/graphs/old/<dataset>.graph` and `datasets/sequences/old/S_<dataset>.seq`.

The name encodes the read: `150_10` is a 150-base read at 10% error, aligned against a 50 000-node graph. `brca2_*` are built from the cactus BRCA2 graph at the same read lengths.

`150_10` and `brca2_150` are the two that matter — both emulate short reads, which is what the report is about. The `5_15`, `20_10`, `30_5` and `150_10_small` pairs are small enough to debug on and are what the verification sweep uses by default.

The raw GFA/FASTQ inputs the `old/` pairs were generated from are in `datasets/graphs/*.gfa` and `datasets/sequences/*.fq`; the loader for those (`read_gfa_graph` / `read_input_sequence`) is in [main.c](main.c) but is currently commented out in `main()`.

### Modes and versions

**Mode 0 — CPU only**

| v | |
|---|---|
| 0 | sequential |
| 1 | SIMD |
| 2 | SIMD, parallel over the DP matrix |
| 3 | SIMD, parallel over the nodes |
| 4 | last column only (the fair baseline against the last-column GPU versions) |
| 5 | last column, multiple reads — set `NUM_READS` |

**Mode 1 — GPU only**

| v | |
|---|---|
| 0 | naive, pageable host graph |
| 1 | naive, pinned host graph |
| 2 | parallel node |
| 3 | parallel async |
| 4 | async monolithic |
| 5 | async batching |
| 7 | shared memory |
| 8 | last column only, traceback recomputed on the CPU |
| 9 | warps — one warp per node, shared memory |
| 10 | registers — one warp per node, shuffles |
| 11 | persistent kernels (registers core, last column only) |
| 12 | registers, halo loads and handover stores merged 32 diagonals at a time |

Version 6 does not exist (it was reserved for a 2-bit encoding) and 13 (short2/DPX) is disabled, as it was not completed either. Versions 8 and up keep one last column per node instead of a full score matrix, so they allocate the graph differently — they are not comparable on wall time with 0-7, only on dense-level throughput.

**Mode 2 — hybrid CPU-GPU**

| v | |
|---|---|
| 0 | base, explicit copies |
| 1 | unified (managed memory) |
| 2 | pinned |
| 3 | advised (managed memory + migration hints) |
| 4 | warps on the dense levels, CPU on the tail (pinned last columns) |
| 5 | registers on the dense levels, CPU on the tail |
| 6 | merged_req on the dense levels, CPU on the tail |
| 7 | short2 — disabled, the dispatch falls through to the warps kernel |

The split point is `HYBRID_MIN_NODES` (default 16): a level with at least that many nodes goes to
the GPU, everything narrower goes to the CPU.

**Mode 3 — no-copy** (only meaningful where CPU and GPU share physical memory, i.e. the DGX Spark)

| v | |
|---|---|
| 0, 1 | naive |
| 2-5 | level |
| 6 | shared memory |
| 7 | last column only |
| 8 | warps |
| 9 | registers |
| 10 | persistent kernels |

The allocator is picked with `SHARED_MEM_KIND` rather than by a version number:

```sh
SHARED_MEM_KIND=advised ./bin/main 150_10 3 6    # managed + preferred location host (default)
SHARED_MEM_KIND=pinned  ./bin/main 150_10 3 6    # cudaMallocHost, pages never move
SHARED_MEM_KIND=managed ./bin/main 150_10 3 6    # plain cudaMallocManaged
```

**Mode 4 — multiple reads**

| v | |
|---|---|
| 0 | warp per (node, read) |
| 1 | same, on the registers core |

```sh
NUM_READS=256 ./bin/main 150_10 4 1              # NUM_READS defaults to 32
```

### Environment variables

| | |
|---|---|
| `NUM_READS` | batch size for modes 0.5 and 4 (default 32) |
| `SHARED_MEM_KIND` | `advised` \| `pinned` \| `managed`, for mode 3 |
| `HYBRID_STATS` | set to anything to make the hybrid and no-copy versions print their launch / wait / CPU split per run |
| `OMP_NUM_THREADS`, `OMP_PLACES` | the usual; on a big.LITTLE part, pinning threads off the small cluster is what stops the CPU phase from wobbling (see `tools/spark_tune.sh`) |

## Tools

All of them are run from the project root and take `-h` for their own usage.

**Correctness.** `tools/bin/verify_dp` runs the CPU sequential aligner and one GPU version on the same input and compares them cell by cell, plus the per-node maxima and the final alignment:

```sh
make -C tools                        # -> tools/bin/verify_dp, tools/bin/spark_probe
tools/bin/verify_dp 500_10 1 9
tools/verify.sh                      # the whole sweep -> outputs/verify.csv
tools/verify.sh -d "30_5 500_10" -v "1:5 1:8" -b "16 64 160"
```

Versions are written `mode:version`. Note that `verify_dp` links only the full-matrix versions — the last-column family (GPU 8-12, hybrid 4-7) is not covered by it.

**Timing and figures.**

```sh
python3 tools/benchmark.py                       # -> outputs/timings.csv
python3 tools/plot_results.py                    # -> outputs/fig_runtime.*, fig_speedup.*
tools/profile.sh                                 # nsys timelines -> outputs/profile/summary.csv
python3 tools/plot_profile.py                    # -> outputs/fig_profile.*
tools/bench_nocopy.sh                            # mode 3 across the three SHARED_MEM_KINDs
```

**The report's numbers.** Nothing in the report is typed by hand; the three scripts below produce every figure and every quoted number:

```sh
python3 tools/report_graph_stats.py              # once, machine independent
python3 tools/report_bench.py -m spark           # on the DGX Spark
python3 tools/report_bench.py -m a30             # on the discrete box
python3 tools/report_figs.py                     # -> report/fig_*.pdf, report/numbers.tex
```

`report_figs.py` draws whatever machines it finds, so it works with only one of the two run.

**Comparison against the state of the art.** `tools/hga_setup.sh` clones and builds HGA (Feng and Luo, ICPP 2021) with the architecture overridden for the machine at hand, and `tools/hga_convert.py` expands one of our segment-level graphs into the base-level CSR HGA expects, so both aligners compute DP matrices of exactly the same size. HGA's sources land under `outputs/`, which is gitignored, so they are never committed here.

**DGX Spark specific.** `tools/spark_check.sh` collects, in one log, everything needed to explain the run-to-run spread of the managed-memory hybrid there; `tools/spark_tune.sh` sweeps `HYBRID_MIN_NODES` and thread placement afterwards. `tools/bin/spark_probe` characterises the machine's memory on its own.

## Layout

```
main.c                  loader, argument handling, verification + timing driver
include/                one header per version, plus definitions.h and the shared *_utils
src/                    one .c/.cu per version
datasets/               graphs/ and sequences/ (the old/ pairs are what bin/main reads)
tools/                  verification, benchmarking, profiling, report and HGA scripts
tests/                  standalone microbenchmarks (copy bandwidth, L2, instruction latency)
utils/                  Albert Jimenez-Blanco's generators for the artificial graphs/sequences
report/                 the report and the presentation
outputs/                everything the scripts write (gitignored)
```
