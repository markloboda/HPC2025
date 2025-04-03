#!/bin/bash

# Check arguments
if [ "$#" -ne 1 ]; then
    echo "Wrong usage: $0 <program>"
    exit 1
fi

PROGRAM=$1
PROGRAM_OUT="bin/$(basename "$PROGRAM" .cu).out"

# Load module CUDA
module load CUDA

# Compile the program
nvcc -diag-suppress 550 -O2 -lm "$PROGRAM" -o "$PROGRAM_OUT"