#!/bin/bash

#SBATCH --job-name=parallel_seam_carving
#SBATCH --output=parallel_seam_carving.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=20000
#SBATCH --reservation=fri

srun ./parallel_seam_carving 1