import re
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class HistogramEqualizationResult:
    mode: str
    image_file: str
    width: int
    height: int
    histogram: float
    cdf: float
    equalize: float
    total_time: float
    sum_all_times: float

def parse_data(data: str) -> List[HistogramEqualizationResult]:
    pattern = re.compile(
        r"--------------- HISTOGRAM EQUALIZATION - (?P<mode>.+?) ---------------\s+"
        r"---------------\s*(?P<image_file>.+?)\s*---------------\s+"
        r"Image width:\s*(?P<width>\d+)\s+"
        r"Image height:\s*(?P<height>\d+)\s+"
        r"Histogram:\s*(?P<histogram>[\d.]+) ms\s+"
        r"CDF:\s*(?P<cdf>[\d.]+) ms\s+"
        r"Equalize:\s*(?P<equalize>[\d.]+) ms\s+"
        r"Total time:\s*(?P<total_time>[\d.]+) ms\s+"
        r"Sum of all times:\s*(?P<sum_all_times>[\d.]+) ms\s+"
        r"[-]+",
        re.MULTILINE
    )
    results = []
    for match in pattern.finditer(data):
        result = HistogramEqualizationResult(
            mode=match.group("mode").strip(),
            image_file=match.group("image_file").strip(),
            width=int(match.group("width")),
            height=int(match.group("height")),
            histogram=float(match.group("histogram")),
            cdf=float(match.group("cdf")),
            equalize=float(match.group("equalize")),
            total_time=float(match.group("total_time")),
            sum_all_times=float(match.group("sum_all_times"))
        )
        results.append(result)
    return results

def remove_worst_runs(results: List[HistogramEqualizationResult], count: int = 3) -> List[HistogramEqualizationResult]:
    """
    Remove the 'count' runs with the highest total_time values.
    If there are less than or equal to count runs, return the list unchanged.
    """
    if len(results) <= count:
        return results
    sorted_runs = sorted(results, key=lambda r: r.total_time)
    return sorted_runs[:-count]

def average_times_per_image(results: List[HistogramEqualizationResult]) -> Dict[str, float]:
    """
    Calculate the average total_time for each unique image_file.
    Returns a dictionary mapping image_file to its average total_time.
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in results:
        sums[r.image_file] += r.total_time
        counts[r.image_file] += 1
    averages = {img: sums[img] / counts[img] for img in sums}
    return averages

def get_image_size(img_filename: str) -> int:
    """
    Extract the image dimensions from the filename (e.g. "720x480")
    and return the total pixel count. If not found, return 0.
    """
    match = re.search(r"(\d+)x(\d+)", img_filename)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width * height
    return 0

def generate_latex_table(averages_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Generate a LaTeX table with algorithms as rows and images as columns.
    The columns (images) are sorted by image size.
    Missing values are shown as 'N/A'.
    """
    image_files = set()
    for algo_averages in averages_dict.values():
        image_files.update(algo_averages.keys())
    image_files = sorted(image_files, key=get_image_size)

    header_columns = "| l ||" + " c |" * len(image_files)
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{header_columns}}}\n"
        "\\hline\n"
        "Algorithm"
    )
    for img in image_files:
        name = img.split("/")[-1].split(".")[0]
        header += f" & {name}"
    header += " \\\\\n\\hline\n"

    rows = []
    for algorithm in sorted(averages_dict.keys()):
        row = algorithm
        algo_averages = averages_dict[algorithm]
        for img in image_files:
            if img in algo_averages:
                row += f" & {algo_averages[img]:.2f}"
            else:
                row += " & N/A"
        row += " \\\\"
        rows.append(row)

    footer = (
        "\n\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Average total times per image for each algorithm after removing the three worst runs.}\n"
        "\\end{table}"
    )
    table = header + "\n".join(rows) + footer
    return table

def compute_speedups(reference: Dict[str, float], other: Dict[str, float]) -> Dict[str, float]:
    """
    Compute speedups comparing a reference set of averages and another set for each image.
    Speedup is defined as (Reference Time) / (Other Time).
    Only images present in both dictionaries are considered.
    """
    speedups = {}
    for img in reference:
        if img in other and other[img] > 0:
            speedups[img] = reference[img] / other[img]
    return speedups

def generate_speedup_comparison_table(comparisons: Dict[str, Dict[str, float]]) -> str:
    """
    Generate a LaTeX table for speedup comparisons.
    Each row corresponds to one comparison (e.g., Parallel/Serial, Parallel Optimized/Serial,
    Parallel Optimized/Parallel).
    Columns represent images (sorted by image size).
    """
    images = set()
    for comp in comparisons.values():
        images.update(comp.keys())
    images = sorted(images, key=get_image_size)

    header_columns = "| l ||" + " c |" * len(images)
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{header_columns}}}\n"
        "\\hline\n"
        "Comparison"
    )
    for img in images:
        name = img.split("/")[-1].split(".")[0]
        header += f" & {name}"
    header += " \\\\\n\\hline\n"

    rows = []
    for label in comparisons.keys():
        row = label
        comp = comparisons[label]
        for img in images:
            if img in comp:
                row += f" & {comp[img]:.2f}"
            else:
                row += " & N/A"
        row += " \\\\"
        rows.append(row)

    footer = (
        "\n\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Speedup comparisons computed as Reference Time divided by Other Time.}\n"
        "\\end{table}"
    )
    table = header + "\n".join(rows) + footer
    return table

def plot_average_times(averages_dict: Dict[str, Dict[str, float]]):
    """
    Plot the average execution times for each algorithm and for each image.
    The x-axis will list the image names (sorted by image size) and each algorithm is plotted as a separate line.
    """
    # Collect and sort all image files
    image_files = set()
    for algo_data in averages_dict.values():
        image_files.update(algo_data.keys())
    image_files = sorted(image_files, key=get_image_size)

    # Prepare x-axis positions and labels
    x = np.arange(len(image_files))
    labels = [img.split("/")[-1].split(".")[0] for img in image_files]

    plt.figure(figsize=(10, 6))

    # Plot each algorithm's average times
    for algorithm, algo_data in averages_dict.items():
        # Get y-values in the order of sorted image files; use np.nan if missing.
        y = [algo_data.get(img, np.nan) for img in image_files]
        plt.plot(x, y, marker='o', label=algorithm)

    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Image")
    plt.ylabel("Average Execution Time (ms)")
    plt.title("Average Execution Time per Image for Each Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.savefig("stats/average_times.png")

def plot_speedups(comparisons: Dict[str, Dict[str, float]]):
    """
    Plot the average speedups for each comparison (each algorithm compared to Serial or to Parallel).
    The x-axis lists the image names (sorted by image size) and each comparison is plotted as a separate line.
    """
    # Collect and sort all image files
    images = set()
    for comp_data in comparisons.values():
        images.update(comp_data.keys())
    images = sorted(images, key=get_image_size)

    x = np.arange(len(images))
    labels = [img.split("/")[-1].split(".")[0] for img in images]

    plt.figure(figsize=(10, 6))

    # Plot each comparison's speedups
    for label, comp_data in comparisons.items():
        y = [comp_data.get(img, np.nan) for img in images]
        plt.plot(x, y, marker='s', label=label)

    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Image")
    plt.ylabel("Speedup")
    plt.title("Speedup Comparisons per Image")
    plt.legend()
    plt.tight_layout()
    plt.savefig("stats/speedup_comparisons.png")

if __name__ == '__main__':
    # File paths for the data files.
    serial_file = "./stats/timing_stats_serial.txt"
    parallel_file = "./stats/timing_stats_parallel.txt"
    optimized_file = "./stats/timing_stats_parallel_optimized.txt"

    # Parse serial data.
    with open(serial_file, "r") as file:
        data = file.read()
    serial_results = parse_data(data)
    serial_filtered = remove_worst_runs(serial_results)
    serial_averages = average_times_per_image(serial_filtered)

    # Parse parallel data.
    with open(parallel_file, "r") as file:
        data = file.read()
    parallel_results = parse_data(data)
    parallel_filtered = remove_worst_runs(parallel_results)
    parallel_averages = average_times_per_image(parallel_filtered)

    # Parse optimized data.
    with open(optimized_file, "r") as file:
        data = file.read()
    optimized_results = parse_data(data)
    optimized_filtered = remove_worst_runs(optimized_results)
    optimized_averages = average_times_per_image(optimized_filtered)

    # Combine averages into one dictionary: algorithm -> { image_file -> avg_time }
    averages_dict = {
        "Serial": serial_averages,
        "Parallel": parallel_averages,
        "Parallel Optimized": optimized_averages,
    }

    # Generate and print the LaTeX table for average times.
    latex_table = generate_latex_table(averages_dict)
    print("\nLaTeX Table (Average Times):\n")
    print(latex_table)

    # Compute speedups compared to Serial.
    parallel_speedups = compute_speedups(serial_averages, parallel_averages)
    optimized_speedups = compute_speedups(serial_averages, optimized_averages)
    # Compute speedups comparing Parallel with Parallel Optimized.
    optimized_vs_parallel = compute_speedups(parallel_averages, optimized_averages)

    # Create a dictionary of comparisons for the speedup table.
    speedup_comparisons = {
        "Parallel/Serial": parallel_speedups,
        "Parallel Optimized/Serial": optimized_speedups,
        "Parallel Optimized/Parallel": optimized_vs_parallel,
    }

    # Generate and print the LaTeX table for speedup comparisons.
    speedup_table = generate_speedup_comparison_table(speedup_comparisons)
    print("\nLaTeX Table (Speedup Comparisons):\n")
    print(speedup_table)

    # Plot average execution times.
    plot_average_times(averages_dict)

    # Plot speedup comparisons.
    plot_speedups(speedup_comparisons)
