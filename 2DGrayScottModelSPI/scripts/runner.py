import subprocess
import os
import sys
from dataclasses import dataclass
from typing import List

PROGRAMS = [
    "gray_scott_model.cpp",
    # "parallel_gray_scott_model.cpp",
]

GRID_SIZES = [
    256,
    # 512,
    # 1024,
    # 2048,
    # 4096,
]

NUM_CORES = [
    1,
    # 2,
    # 4,
    # 16,
    # 32,
    # 64
]

NUM_RUNS = 1

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
        compiled_program = f"./bin/{job.program.replace('.cpp', '.out')}"
        cmd = ["sbatch", "scripts/job.sh", compiled_program, str(job.grid_size), str(job.num_of_cores)]
        print("Submitting with sbatch:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=".", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())

def run_with_srun(jobs: List[SlurmJob]):
    for job in jobs:
        compiled_program = f"./bin/{job.program.replace('.cpp', '.out')}"
        cmd = [
            "srun",
            "--job-name=runner-run-gray-scott-model",
            "--nodes=1",
            "--ntasks-per-node=1",
            "--cpus-per-task=1",
            "--threads-per-core=1",
            "--mem-per-cpu=2G",
            "--time=10:00",
            "--output=logs/runner-run-gray-scott-model.log",
            "--reservation=fri",
            compiled_program,
            str(job.grid_size),
            str(job.num_of_cores)
        ]
        print("Running with srun:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=".", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        if result.returncode != 0:
            print("Error:", result.stderr.decode())

if __name__ == "__main__":
    use_srun = "--use-srun" in sys.argv

    print("runner.py started")

    jobs = []
    for program in PROGRAMS:
        for grid_size in GRID_SIZES:
            for num_of_cores in NUM_CORES:
                for i in range(NUM_RUNS):
                    jobs.append(SlurmJob(program, grid_size, num_of_cores))

    original_dir = os.getcwd()
    os.chdir("..")

    print("Compiling required programs...")
    try:
        compile_programs()
    except Exception as e:
        print(f"Compilation failed: {e}")
        exit(1)

    print("Running jobs...")

    if use_srun:
        run_with_srun(jobs)
    else:
        run_slurm_jobs(jobs)

    os.chdir(original_dir)
