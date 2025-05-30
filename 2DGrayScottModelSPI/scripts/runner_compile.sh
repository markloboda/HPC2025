#!/bin/bash

# Check arguments
if [ "$#" -ne 1 ]; then
    echo "Wrong usage: $0 <program>"
    exit 1
fi

PROGRAM=$1
PROGRAM_OUT="bin/$(basename "$PROGRAM" .cu).out"

mkdir -p bin

module load OpenMPI

# Compile the program
mpicc -o $PROGRAM_OUT $PROGRAM
