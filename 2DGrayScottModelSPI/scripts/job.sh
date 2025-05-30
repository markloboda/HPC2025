
#!/bin/bash

#SBATCH --job-name=runner-run-gray-scott-model
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00
#SBATCH --output=logs/runner-run-gray-scott-model.log
#SBATCH --reservation=fri

module load OpenMPI

PROGRAM=$1
GRID_SIZE=$2

echo "Running program: $PROGRAM with grid size: $GRID_SIZE"
mpirun --display-allocation -n 8 $SLURM_JOB_NAME $GRID_SIZE