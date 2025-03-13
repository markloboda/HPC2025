#!/bin/bash

#SBATCH --job-name=seam_carving
#SBATCH --output=seam_carving.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=20000
#SBATCH --reservation=fri

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compiling
gcc -O2 -lm --openmp seam_carving.c -o seam_carving.out

# Run
srun ./seam_carving.out ./test_images/720x480.png ./output_images/720x480.png 720
