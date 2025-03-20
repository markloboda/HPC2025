#!/bin/bash

#SBATCH --job-name=seam_carving_parallel
#SBATCH --output=seam_carving_parallel.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=20000
#SBATCH --reservation=fri

# export OMP_PLACES=cores
# export OMP_PROC_BIND=TRUE
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compiling
gcc -O2 -lm --openmp ../parallel_seam_carving.c -o ../parallel_seam_carving.out

# Run
srun ../parallel_seam_carving.out ../test_images/1024x768.png ../output_images/1024x768_parallel.png 128
