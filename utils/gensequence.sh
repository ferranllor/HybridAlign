# 1. Simulate reads and output their alignments (.gam)
./vg sim -n 1000 -l 4500 -x ../datasets/graphs/"$1".gfa -a > "$2".gam

# 2. Extract standard FASTQ sequences
./vg view -X "$2".gam > "$2".fq

# 3. Extract Read ID and the Ground Truth Sequence into a TSV file
./vg view -aj "$2".gam | jq -r '[.name, .sequence] | @tsv' > "$2".tsv
