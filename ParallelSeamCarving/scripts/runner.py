import subprocess
import os
from dataclasses import dataclass
from typing import List

NUM_THREADS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]

PROGRAMS = [
   "seam_carving.c",
    "seam_carving_optimized.c",
    "parallel_seam_carving.c",
    "parallel_seam_carving_triangles.c",
     "parallel_seam_carving_triangles_greedy.c",
]

IMAGES = [
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
    cpus: int
    program: str
    input_image: str
    output_image: str
    num_seams: int

def compile_programs():
    # Run program runner_compile.sh with arguments: PROGRAM=$1 CPUS=$2
    for program in PROGRAMS:
        cmd = ["./scripts/runner_compile.sh", program]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
        print("Done compiling", program)

def run_slurm_jobs(jobs: List[SlurmJob], script_path: str):
    for job in jobs:
        compiled_program = f"./bin/{job.program.replace('.c', '.out')}"

        cmd = ["srun",
               "--job-name=runner-run-seam_carving",
               "--output=logs/runner-run-seam_carving.log",
               "--ntasks=1",
               "--nodes=1",
               f"--cpus-per-task={job.cpus}",
               "--time=60:00",
               "--reservation=fri",
               compiled_program,
               job.input_image,
               job.output_image,
               str(job.num_seams)]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=".", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = result.stdout.decode()
        print(result)

if __name__ == "__main__":
    print("runner.py started")

    jobs = []
    for image_index in range(len(IMAGES)):
        for program in PROGRAMS:
            if (program == "seam_carving.c" or program == "seam_carving_optimized.c"):
                image = IMAGES[image_index]
                outs = OUT_IMAGES[image_index].split(".")
                out_image = f"{outs[0]}_{program}_{1}.{outs[1]}"
                for i in range(NUM_RUNS):
                    jobs.append(SlurmJob(1, program, image, out_image, 128))
            else:
                for num_threads in NUM_THREADS:
                    image = IMAGES[image_index]
                    outs = OUT_IMAGES[image_index].split(".")
                    out_image = f"{outs[0]}_{program}_{num_threads}.{outs[1]}"
                    for i in range(NUM_RUNS):
                        jobs.append(SlurmJob(num_threads, program, image, out_image, 128))

    print(jobs)

    original_dir = os.getcwd()
    os.chdir("..")

    print("Compiling required programs...")
    compile_programs()

    print("Running jobs...")
    run_slurm_jobs(jobs, "scripts/runner_run.sh")

    os.chdir(original_dir)