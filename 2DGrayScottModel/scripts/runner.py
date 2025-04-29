import subprocess
import os
from dataclasses import dataclass
from typing import List

PROGRAMS = [
    "gray_scott_model.cu",
    "parallel_gray_scott_model.cu",
    "parallel_gray_scott_model_optimized.cu",
]

GRID_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
]

NUM_RUNS = 5

@dataclass
class SlurmJob:
    program: str
    grid_size: int

def compile_programs():
    # Run program runner_compile.sh with arguments: PROGRAM=$1
    for program in PROGRAMS:
        cmd = ["./scripts/runner_compile.sh", program]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
        print("Done compiling", program)

def run_slurm_jobs(jobs: List[SlurmJob]):
    for job in jobs:
        compiled_program = f"./bin/{job.program.replace('.cu', '.out')}"

        cmd = ["sbatch", "scripts/job.sh", compiled_program, str(job.grid_size)]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=".", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = result.stdout.decode()
        print(result)

if __name__ == "__main__":
    print("runner.py started")

    jobs = []
    for program in PROGRAMS:
        cache_run = True  # First run on GPU is slower due to caching of kernels
        for grid_size_index in range(len(GRID_SIZES)):
            grid_size = GRID_SIZES[grid_size_index]
            for i in range(int(cache_run) + NUM_RUNS):
                cache_run = False
                jobs.append(SlurmJob(program, grid_size))

    # print(jobs)

    original_dir = os.getcwd()
    os.chdir("..")

    print("Compiling required programs...")
    compile_programs()

    print("Running jobs...")
    run_slurm_jobs(jobs)

    os.chdir(original_dir)
