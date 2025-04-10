#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=runner-run-histogram_equalization
#SBATCH --output=logs/runner-run-histogram_equalization.log
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --reservation=fri

PROGRAM=$1
IMAGE_IN=$2
IMAGE_OUT=$3

echo "Running program: $PROGRAM with input: $IMAGE_IN and output: $IMAGE_OUT"
srun $PROGRAM $IMAGE_IN $IMAGE_OUT