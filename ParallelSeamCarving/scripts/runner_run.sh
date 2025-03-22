#!/bin/bash
export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <cpus> <program> <input_image> <output_image> <num_seams>"
    exit 1
fi

CPUS=$1
PROGRAM=$2
INPUT_IMAGE=$3
OUTPUT_IMAGE=$4
NUM_SEAMS=$5

PROGRAM_OUT="bin/$(basename "$PROGRAM" .c).out"

# Run
echo "Running $PROGRAM_OUT $INPUT_IMAGE $OUTPUT_IMAGE $NUM_SEAMS"
./$PROGRAM_OUT "$INPUT_IMAGE" "$OUTPUT_IMAGE" $NUM_SEAMS
