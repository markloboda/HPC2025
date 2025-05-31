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

# Example usage:
if __name__ == "__main__":
    serial_filename = '../timing_stats/timing_stats_serial.txt'
    parallel_filename = '../timing_stats/timing_stats_parallel.txt'
    optimized_filename = '../timing_stats/timing_stats_optimized.txt'
    serial_averages = parse_histogram_file(serial_filename )
    parallel_averages = parse_histogram_file(parallel_filename )
    optimized_averages = parse_histogram_file(optimized_filename )

    print(f"Serial")
    for grid_size, avg_time in sorted(serial_averages.items()):
        print(f"Grid Size: {grid_size}, Average Time (ms): {avg_time:.6f}")
    
    print()
    print(f"Parallel")
    for grid_size, avg_time in sorted(parallel_averages.items()):
        print(f"Grid Size: {grid_size}, Average Time (ms): {avg_time:.6f}")

    print()
    print(f"Parallel Optimized")
    for grid_size, avg_time in sorted(optimized_averages.items()):
        print(f"Grid Size: {grid_size}, Average Time (ms): {avg_time:.6f}")