#!/bin/bash

#SBATCH --job-name=seam_carving
#SBATCH --output=seam_carving.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=5:00
#SBATCH --mem-per-cpu=20000
#SBATCH --reservation=fri

srun ./seam_carving 1