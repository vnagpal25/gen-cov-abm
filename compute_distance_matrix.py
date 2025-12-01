#!/usr/bin/env python3
"""
Compute evolutionary distance matrix for nodes in ma_sequences.csv
based on phylogenetic tree structure in all_sequences.csv

This script uses CSV data rather than loading the full tree structure,
but follows the same logic as the PhylogeneticGraph class methods.
"""

import numpy as np
from typing import Dict, List
import time
from tqdm import trange


class CSVPhylogeneticTree:
    """
    Lightweight phylogenetic tree structure built from CSV data.

    optimized for computing distance matrices without loading
    the full CovidGenomeSequence tree structure.
    """

    def __init__(self, parent_map: Dict[str, str], divergence_map: Dict[str, float]):
        """
        Initialize the tree structure.

        Args:
            parent_map: Maps node name to parent node name
            divergence_map: Maps node name to divergence value from root
        """
        self.parent_map = parent_map
        self.divergence_map = divergence_map

    def get_path_to_root(self, node_name: str) -> List[str]:
        """Get the path from a node to the root."""
        path = [node_name]  # path to root must include the node itself
        current = node_name
        while current in self.parent_map:
            current = self.parent_map[current]
            path.append(current)
        return path

    def find_lca(self, node1: str, node2: str) -> str:
        """Find the Lowest Common Ancestor of two nodes."""
        path1 = self.get_path_to_root(node1)
        path2 = self.get_path_to_root(node2)

        # print(f'{node1}: {path1}')
        # print(f'{node2}: {path2}')

        # Convert path2 to set for efficient lookup
        path2_set = set(path2)

        # Find the first (deepest) common ancestor
        for node in path1:
            if node in path2_set:
                return node

        return None

    def compute_evolutionary_distance(self, node1: str, node2: str) -> float:
        """
        Compute evolutionary distance between two nodes.

        The distance is calculated as the sum of divergence differences along the path
        connecting the two nodes through their Lowest Common Ancestor (LCA).

        Args:
            node1: First node name
            node2: Second node name

        Returns:
            Evolutionary distance (sum of divergence differences)
        """
        if node1 == node2:
            return 0.0

        lca = self.find_lca(node1, node2)
        if lca is None:
            raise Exception("Nodes don't have a common ancestor")

        # Compute distance as sum of divergence differences through LCA
        distance = abs(self.divergence_map[node1] - self.divergence_map[lca]) + abs(
            self.divergence_map[node2] - self.divergence_map[lca]
        )

        return distance

    def compute_distance_matrix(self, node_names: List[str]) -> np.ndarray:
        """
        Compute an n × n evolutionary distance matrix for a list of nodes.

        Args:
            node_names: List of node names to compute distances for
            verbose: If True, print progress updates

        Returns:
            n × n NumPy array of pairwise distances
        """
        n = len(node_names)

        # Initialize distance matrix
        distance_matrix = np.zeros((n, n))
        start_time = time.time()

        # Compute pairwise distances
        # only upper triangular to repeat unecessary computation
        for i in trange(n):
            for j in range(i, n):
                dist = self.compute_evolutionary_distance(node_names[i], node_names[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix
