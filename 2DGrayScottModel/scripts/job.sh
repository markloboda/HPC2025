#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=runner-run-gray-scott-model
#SBATCH --output=logs/runner-run-gray-scott-model.log
#SBATCH --gpus=2
#SBATCH --time=00:10:00
#SBATCH --reservation=fri

PROGRAM=$1
GRID_SIZE=$2

echo "Running program: $PROGRAM with grid size: $GRID_SIZE"
srun $PROGRAM $GRID_SIZE