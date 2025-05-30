import re
from collections import defaultdict

def parse_histogram_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Match each block of data
    pattern = re.compile(
        r'Grid size:\s*(\d+)\s*Total time:\s*([\d.]+)\s*ms',
        re.MULTILINE
    )

    # Extract all (grid_size, time) pairs
    data = defaultdict(list)
    for match in pattern.finditer(content):
        grid_size = int(match.group(1))
        time = float(match.group(2))
        data[grid_size].append(time)

    # Process data: discard 3 worst times and calculate average
    results = {}
    for grid_size, times in data.items():
        times.sort()  # ascending order
        if len(times) > 3:
            trimmed_times = times[:-3]  # discard 3 worst (largest)
        else:
            trimmed_times = times  # not enough to discard
        avg_time = sum(trimmed_times) / len(trimmed_times)
        results[grid_size] = avg_time

    return results

def compute_speedup(serial_times, other_times):
    speedups = {}
    for grid_size, serial_time in serial_times.items():
        if grid_size in other_times:
            speedup = serial_time / other_times[grid_size]
            speedups[grid_size] = speedup
    return speedups

# Example usage:
if __name__ == "__main__":
    serial_filename = 'timing_stats_serial.txt'
    parallel_filename = 'timing_stats_parallel.txt'
    optimized_1_filename = 'timing_stats_optimized_1_gpu.txt'
    optimized_2_filename = 'timing_stats_optimized_2_gpu.txt'

    serial_averages = parse_histogram_file(serial_filename)
    parallel_averages = parse_histogram_file(parallel_filename)
    optimized_1_averages = parse_histogram_file(optimized_1_filename)
    optimized_2_averages = parse_histogram_file(optimized_2_filename)

    # Speedups
    parallel_speedups = compute_speedup(serial_averages, parallel_averages)
    optimized_1_speedups = compute_speedup(serial_averages, optimized_1_averages)
    optimized_2_speedups = compute_speedup(serial_averages, optimized_2_averages)

    print(f"Serial")
    for grid_size, avg_time in sorted(serial_averages.items()):
        print(f"Grid Size: {grid_size}, Average Time (ms): {avg_time:.2f}")
    
    print(f"\nParallel")
    for grid_size, avg_time in sorted(parallel_averages.items()):
        speedup = parallel_speedups.get(grid_size, None)
        print(f"Grid Size: {grid_size}, Average Time (ms): {avg_time:.2f}", end='')
        if speedup:
            print(f", Speedup: {speedup:.2f}")
        else:
            print()

    print(f"\nParallel Optimized with 1 GPU")
    for grid_size, avg_time in sorted(optimized_1_averages.items()):
        speedup = optimized_1_speedups.get(grid_size, None)
        print(f"Grid Size: {grid_size}, Average Time (ms): {avg_time:.2f}", end='')
        if speedup:
            print(f", Speedup: {speedup:.2f}")
        else:
            print()

    print(f"\nParallel Optimized with 2 GPUs")
    for grid_size, avg_time in sorted(optimized_2_averages.items()):
        speedup = optimized_2_speedups.get(grid_size, None)
        print(f"Grid Size: {grid_size}, Average Time (ms): {avg_time:.2f}", end='')
        if speedup:
            print(f", Speedup: {speedup:.2f}")
        else:
            print()
