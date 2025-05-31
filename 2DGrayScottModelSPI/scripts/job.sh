#!/bin/bash

#SBATCH --job-name=runner-run-gray-scott-model
#SBATCH --output=logs/runner-run-gray-scott-model.out
#SBATCH --error=logs/runner-run-gray-scott-model.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00
#SBATCH --reservation=fri

module load OpenMPI

PROGRAM=$1
GRID_SIZE=$2
NUM_CORES=$3

echo "Running program: $PROGRAM with grid size: $GRID_SIZE and number of cores: $NUM_CORES"
mpirun -np "$NUM_CORES" "$PROGRAM" "$GRID_SIZE" "$NUM_CORES"