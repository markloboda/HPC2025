import subprocess
import os
from dataclasses import dataclass
from typing import List

PROGRAMS = [
    "histogram_equalization.cu",
    "parallel_histogram_equalization.cu"
]

IN_IMAGES = [
    "test_images/720x480.png",
    "test_images/1024x768.png",
    "test_images/1920x1200.png",
    "test_images/3840x2160.png",
    "test_images/7680x4320.png",
]

OUT_IMAGES = [
    "output_images/720x480.png",
    "output_images/1024x768.png",
    "output_images/1920x1200.png",
    "output_images/3840x2160.png",
    "output_images/7680x4320.png",
]

NUM_RUNS = 1

@dataclass
class SlurmJob:
    program: str
    input_image: str
    output_image: str

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

        cmd = ["srun",
               "--partition=gpu",
               "--job-name=runner-run-histogram_equalization",
               "--output=logs/runner-run-histogram_equalization.log",
               "--gpus=1",
               "--time=60:00",
               "--reservation=fri",
               compiled_program,
               job.input_image,
               job.output_image]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=".", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = result.stdout.decode()
        print(result)

if __name__ == "__main__":
    print("runner.py started")

    jobs = []
    for image_index in range(len(IN_IMAGES)):
        for program in PROGRAMS:
            image = IN_IMAGES[image_index]
            outs = OUT_IMAGES[image_index].split(".")
            program_name = program.replace(".cu", "")
            out_image = f"{outs[0]}_{program_name}.{outs[1]}"
            for i in range(NUM_RUNS):
                jobs.append(SlurmJob(program, image, out_image))

    # print(jobs)

    original_dir = os.getcwd()
    os.chdir("..")

    print("Compiling required programs...")
    compile_programs()

    print("Running jobs...")
    run_slurm_jobs(jobs)

    os.chdir(original_dir)