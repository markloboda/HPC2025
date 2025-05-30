import subprocess
import os
from dataclasses import dataclass
from typing import List

PROGRAMS = [
    "gray_scott_model.cu",
    "parallel_gray_scott_model.cu",
]

GRID_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
]

NUM_CORES = [
    1,
    2,
    4,
    16,
    32,
    64
]

NUM_RUNS = 8

@dataclass
class SlurmJob:
    program: str
    grid_size: int
    num_of_cores: int

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

        cmd = ["sbatch", "scripts/job.sh", compiled_program, str(job.grid_size), str(job.num_of_cores)]
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
            for num_of_cores_index in range(len(NUM_CORES)):
                num_of_cores = NUM_CORES[num_of_cores_index]
                for i in range(NUM_RUNS):
                    jobs.append(SlurmJob(program, grid_size, num_of_cores))

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
