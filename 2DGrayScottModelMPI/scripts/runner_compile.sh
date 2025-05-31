#!/bin/bash

# Check arguments
if [ "$#" -ne 1 ]; then
    echo "Wrong usage: $0 <program.cpp>"
    exit 1
fi

PROGRAM=$1
EXT="${PROGRAM##*.}"
BASENAME="$(basename "$PROGRAM" .${EXT})"
PROGRAM_OUT="bin/${BASENAME}.out"

mkdir -p bin

module load OpenMPI

# Choose compiler based on file extension
if [ "$EXT" = "cpp" ] || [ "$EXT" = "cc" ] || [ "$EXT" = "cxx" ]; then
    mpic++ -o "$PROGRAM_OUT" "$PROGRAM"
elif [ "$EXT" = "c" ]; then
    mpicc -o "$PROGRAM_OUT" "$PROGRAM"
else
    echo "Unsupported file extension: .$EXT"
    exit 1
fi
