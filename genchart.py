import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

methods = [
    'pgvector_ivfflat_halfvec_l2_ops',
    'pgvector_hnsw_halfvec_l2_ops',
    'pgvector_hnsw_vector_l2_ops',
    'pgvector_ivfflat_vector_l2_ops',
    'os_nmslib',
    'os_lucene',
    'os_faiss'
]

store_time = [5.68, 5.47, 5.32, 8.96, 10.83, 10.44, 13.75]
query_time = [2.65, 2.69, 2.70, 3.43, 5.88, 6.01, 6.52]
precision = [0.9992, 1.0000, 0.9992, 1.0000, 0.9936, 0.9938, 0.9944]

x = np.arange(len(methods))
width = 0.3  # Increased width for better visibility

fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot store time and query time on the primary y-axis
rects1 = ax1.bar(x - width/2, store_time, width, label='Store Time (s)', color='skyblue')
rects2 = ax1.bar(x + width/2, query_time, width, label='Query Time (s)', color='lightgreen')

ax1.set_ylabel('Time (seconds)')
ax1.set_title('Benchmark Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha='right')

# Create a secondary y-axis for precision
ax2 = ax1.twinx()
rects3 = ax2.bar(x, precision, width, label='Precision', color='salmon', alpha=0.7)

ax2.set_ylabel('Precision')
ax2.set_ylim(0, 1.1)  # Set y-limit for precision axis

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()

# Generate filename with current date
current_date = datetime.now().strftime("%Y-%m-%d")
filename = f"benchmark_comparison_{current_date}.png"

# Save the figure
plt.savefig(filename, dpi=300, bbox_inches='tight')

print(f"Image saved as {filename}")
