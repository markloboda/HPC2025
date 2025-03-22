#!/bin/bash
export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check arguments
if [ "$#" -ne 1 ]; then
    echo "Wrong usage: $0 <program>"
    exit 1
fi

PROGRAM=$1
PROGRAM_OUT="bin/$(basename "$PROGRAM" .c).out"

# Compile the program
gcc -O0 -lm --openmp "$PROGRAM" -o "$PROGRAM_OUT"