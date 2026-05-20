# 1. Simulate reads and output their alignments (.gam)
./vg sim -n 1000 -l 150 -x ../HybridAlign/datasets/graphs/"$1".gfa -a > "$1".gam

# 2. Extract standard FASTQ sequences
./vg view -X "$1".gam > "$1".fq

# 3. Extract Read ID and the Ground Truth Sequence into a TSV file
./vg view -aj "$1".gam | jq -r '[.name, .sequence] | @tsv' > "$1".tsv
