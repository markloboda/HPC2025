import re
from collections import defaultdict
import os

def parse_log(file_path, idx):
    data = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r') as f:
        content = f.read()

    splitters = [
        "--------------- PARALLEL SEAM CARVING TRIANGLES GREEDY ---------------",
        "--------------- PARALLEL SEAM CARVING TRIANGLES ---------------",
        "--------------- PARALLEL SEAM CARVING ---------------",
        "--------------- SEAM CARVING SEQUENTIAL OPTIMIZED ---------------",
        "--------------- SEAM CARVING SEQUENTIAL ---------------",
    ]

    entries = content.split(splitters[idx])

    for entry in entries:
        if not entry.strip():
            continue

        image_match = re.search(r'--------------- (.+?) ---------------', entry)
        cpu_match = re.search(r'CPUs: (\d+)', entry)
        seam_match = re.search(r'seamCount=(\d+)', entry)
        time_match = re.search(r'Total Processing Time: ([\d.]+)s', entry)
        energy_match = re.search(r'Energy Calculations: ([\d.]+)s', entry)
        seam_id_match = re.search(r'Seam Identifications: ([\d.]+)s', entry)
        seam_annotate_match = re.search(r'Seam Annotates: ([\d.]+)s', entry)
        seam_remove_match = re.search(r'Seam Removes: ([\d.]+)s', entry)

        if image_match and cpu_match and seam_match and time_match:
            image_name = image_match.group(1)
            cpu_count = int(cpu_match.group(1))
            seam_count = int(seam_match.group(1))
            total_time = float(time_match.group(1))
            energy_time = float(energy_match.group(1)) if energy_match else 0
            seam_id_time = float(seam_id_match.group(1)) if seam_id_match else 0
            seam_annotate_time = float(seam_annotate_match.group(1)) if seam_annotate_match else 0
            seam_remove_time = float(seam_remove_match.group(1)) if seam_remove_match else 0

            key = (image_name, cpu_count, seam_count)
            data[key]['total_time'].append(total_time)
            data[key]['energy_time'].append(energy_time)
            data[key]['seam_id_time'].append(seam_id_time)
            data[key]['seam_annotate_time'].append(seam_annotate_time)
            data[key]['seam_remove_time'].append(seam_remove_time)

    return data

def compute_averages(data):
    averages = {}
    for key, values in data.items():
        avg_total_time = sum(values['total_time']) / len(values['total_time'])
        avg_energy_time = sum(values['energy_time']) / len(values['energy_time'])
        avg_seam_id_time = sum(values['seam_id_time']) / len(values['seam_id_time'])
        avg_seam_annotate_time = sum(values['seam_annotate_time']) / len(values['seam_annotate_time'])
        avg_seam_remove_time = sum(values['seam_remove_time']) / len(values['seam_remove_time'])

        averages[key] = {
            'avg_total_time': avg_total_time,
            'avg_energy_time': avg_energy_time,
            'avg_seam_id_time': avg_seam_id_time,
            'avg_seam_annotate_time': avg_seam_annotate_time,
            'avg_seam_remove_time': avg_seam_remove_time,
        }
    return averages

def save_results(file_path, averages):
    output_dir = "main_run/parsed/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_parsed.txt")

    with open(output_file, 'w') as f:
        for (image, cpu, seam), stats in sorted(averages.items()):
            f.write(f"Image: {image}, CPUs: {cpu}, Seam Count: {seam},\n")
            f.write(f"  Avg Total Time: {stats['avg_total_time']:.6f}s\n")
            f.write(f"  Avg Energy Calculations: {stats['avg_energy_time']:.6f}s\n")
            f.write(f"  Avg Seam Identifications: {stats['avg_seam_id_time']:.6f}s\n")
            f.write(f"  Avg Seam Annotates: {stats['avg_seam_annotate_time']:.6f}s\n")
            f.write(f"  Avg Seam Removes: {stats['avg_seam_remove_time']:.6f}s\n\n")

def main():
    file_paths = [
        "main_run/raw/timing_stats_parallel_triangles_greedy.txt",
        "main_run/raw/timing_stats_parallel_triangles.txt",
        "main_run/raw/timing_stats_parallel.txt",
        "main_run/raw/timing_stats_sequential_optimized.txt",
        "main_run/raw/timing_stats_sequential.txt",
    ]

    for file_path_idx in range(len(file_paths)):
        parsed_data = parse_log(file_paths[file_path_idx], file_path_idx)
        averages = compute_averages(parsed_data)
        save_results(file_paths[file_path_idx], averages)
        print(f"Results saved to /main_run/parsed/{os.path.splitext(os.path.basename(file_paths[file_path_idx]))[0]}_parsed.txt")

if __name__ == "__main__":
    main()
