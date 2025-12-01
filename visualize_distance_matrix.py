#!/usr/bin/env python3
"""
Visualize the evolutionary distance matrix
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Load the distance matrix
print("Loading distance matrix...")
distance_matrix = np.load('data/ma_evolutionary_distance_matrix.npy')
print(f"Matrix shape: {distance_matrix.shape}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Heatmap of a subset of the matrix
sample_size = min(200, distance_matrix.shape[0])
sample_matrix = distance_matrix[:sample_size, :sample_size]

ax = axes[0, 0]
im = ax.imshow(sample_matrix, cmap='viridis', aspect='auto')
ax.set_title(f'Distance Matrix Heatmap (first {sample_size} nodes)')
ax.set_xlabel('Node index')
ax.set_ylabel('Node index')
plt.colorbar(im, ax=ax, label='Evolutionary distance')

# 2. Distribution of distances
ax = axes[0, 1]
upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
distances = distance_matrix[upper_tri_indices]
finite_distances = distances[np.isfinite(distances)]

ax.hist(finite_distances, bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Evolutionary distance')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Pairwise Evolutionary Distances')
ax.axvline(np.mean(finite_distances), color='red', linestyle='--', 
           label=f'Mean: {np.mean(finite_distances):.2f}')
ax.axvline(np.median(finite_distances), color='orange', linestyle='--', 
           label=f'Median: {np.median(finite_distances):.2f}')
ax.legend()
ax.grid(alpha=0.3)

# 3. Cumulative distribution
ax = axes[1, 0]
sorted_distances = np.sort(finite_distances)
cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
ax.plot(sorted_distances, cumulative, linewidth=2)
ax.set_xlabel('Evolutionary distance')
ax.set_ylabel('Cumulative probability')
ax.set_title('Cumulative Distribution of Evolutionary Distances')
ax.grid(alpha=0.3)

# 4. Statistics summary
ax = axes[1, 1]
ax.axis('off')

stats_text = f"""
Evolutionary Distance Matrix Statistics

Matrix size: {distance_matrix.shape[0]} × {distance_matrix.shape[1]}
Total pairwise distances: {len(finite_distances):,}

Distance statistics (off-diagonal):
  Min:     {np.min(finite_distances):.2f}
  Q1:      {np.percentile(finite_distances, 25):.2f}
  Median:  {np.median(finite_distances):.2f}
  Q3:      {np.percentile(finite_distances, 75):.2f}
  Max:     {np.max(finite_distances):.2f}
  Mean:    {np.mean(finite_distances):.2f}
  Std:     {np.std(finite_distances):.2f}

Files saved:
  • ma_evolutionary_distance_matrix.npy
  • ma_evolutionary_distance_matrix.csv
  • ma_distance_matrix_node_names.txt
"""

ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
        verticalalignment='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('data/ma_evolutionary_distance_visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: data/ma_evolutionary_distance_visualization.png")
plt.close()

print("\nDone!")

