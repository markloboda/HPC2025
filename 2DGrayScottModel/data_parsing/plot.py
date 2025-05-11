import matplotlib.pyplot as plt

# Data
grid_sizes = [256, 512, 1024, 2048, 4096]

serial_times = [4727.29, 13884.00, 50175.90, 193547.03, 764905.96]
parallel_times = [33.43, 42.22, 141.33, 488.38, 1984.08]
parallel_opt_1gpu = [26.38, 32.69, 117.11, 401.34, 1531.17]
parallel_opt_2gpu = [610.50, 621.10, 671.84, 874.95, 1565.30]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(grid_sizes, serial_times, marker='o', label='Serial')
plt.plot(grid_sizes, parallel_times, marker='o', label='Parallel')
plt.plot(grid_sizes, parallel_opt_1gpu, marker='o', label='Parallel Optimized (1 GPU)')
plt.plot(grid_sizes, parallel_opt_2gpu, marker='o', label='Parallel Optimized (2 GPUs)')

# Log scale for better visibility
plt.yscale('log')

plt.title('Average Execution Time per Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('Average Time (ms) [Log Scale]')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()

# Save the plot
plt.savefig('execution_time_comparison.png', dpi=300)
plt.show()