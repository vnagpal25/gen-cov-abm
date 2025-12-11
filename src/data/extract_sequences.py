#!/usr/bin/env python
"""
Extract sequences from phylogenetic tree data.

This script loads a phylogenetic tree from JSON files and generates
genome sequences for specified proteins and outputs to a CSV file.
"""

import sys
import json
from pathlib import Path

# Add src directory to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))

from data.genome_node import CovidGenomeSequence
from data.tree_to_csv import tree_to_csv
from utils.path_utils import get_data_dir


def main():
    
    print("Loading tree data...")
    data_dir = get_data_dir("genomic_data")
    tree_path = data_dir / "tree.json"
    root_seq_path = data_dir / "root-sequence.json"
    
    with open(tree_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    
    with open(root_seq_path, "rt", encoding="utf-8") as f:
        root_seq = json.load(f)
    
    # extract phylogenetic tree
    tree = data['tree']
    
    print("Processing tree and generating sequences...")
    # Create tree object and generate sequences for specified proteins
    tree_obj = CovidGenomeSequence(tree)
    tree_obj.generate_sequences(root_seq, proteins=["N", "S", "ORF1a"])
    
    # Write sequences to CSV
    output_filename = "all_sequences.csv"
    tree_to_csv(tree_obj, output_filename)
    
    print(f"Sequences written to {output_filename}. Done.")

if __name__ == "__main__":
    main()
