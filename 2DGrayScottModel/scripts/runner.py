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

NUM_RUNS = 8

@dataclass
class SlurmJob:
    program: str
    grid_size: int

def compile_programs():
    for program in PROGRAMS:
        cmd = ["./scripts/runner_compile.sh", program]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error compiling {program}: {result.stderr.decode()}")
            raise RuntimeError(f"Compilation failed for {program}")
        print("Successfully compiled", program)

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
        for grid_size_index in range(len(GRID_SIZES)):
            grid_size = GRID_SIZES[grid_size_index]
            for i in range(NUM_RUNS):
                jobs.append(SlurmJob(program, grid_size))

    # print(jobs)

    original_dir = os.getcwd()
    os.chdir("..")

    print("Compiling required programs...")
    try:
        compile_programs()
    except Exception as e:
        print(f"Compilation failed: {e}")
        exit(1)

    print("Running jobs...")



    run_slurm_jobs(jobs)

    os.chdir(original_dir)
