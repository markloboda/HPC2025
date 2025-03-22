#!/bin/bash
#SBATCH --job-name=runner
#SBATCH --output=../logs/runner.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --reservation=fri

echo "Starting runner.py"
python3 runner.py