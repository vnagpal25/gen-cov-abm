import csv
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm


from genome_node import CovidGenomeSequence
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.path_utils import get_data_dir


def tree_to_csv(
    root: CovidGenomeSequence,
    output_file: str,
    children_attr: str = "children",
):
    """
    Convert genome tree to CSV format.

    Args:
        root: Root object of the tree
        output_file: Path to output CSV file
        children_attr: Name of the attribute containing children (default: 'children')
    """
    # Collect all objects in the tree
    all_objects = []
    output_path = get_data_dir() / 'genomic_data' / Path(output_file)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Progress bar for traversal
    pbar = tqdm(desc="Traversing tree", unit=" nodes")
    
    def traverse(obj, depth=0, parent: str = None):
        """Recursively traverse the tree and collect objects."""
        # Add current object with metadata
        obj_data = {"_depth": depth, "_parent_name": parent}

        # Get all primitive attributes except children and mutations
        for attr, value in vars(obj).items():
            if attr != children_attr and type(value) in [str, int, float, bool]:
                obj_data[attr] = value

        all_objects.append(obj_data)
        pbar.update(1)

        # Traverse children
        children = getattr(obj, children_attr, None)
        if children:
            for child in children:
                traverse(child, depth + 1, obj_data["name"])

    traverse(root)
    pbar.close()

    # Get all unique field names
    fieldnames = set()
    for obj_data in all_objects:
        fieldnames.update(obj_data.keys())

    # Sort fieldnames for consistent output (put metadata fields first)
    metadata_fields = ["_parent_name", "_depth"]
    other_fields = sorted(fieldnames - set(metadata_fields))
    fieldnames = metadata_fields + other_fields

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write rows with progress bar
        for row in tqdm(all_objects, desc="Writing CSV", unit=" rows"):
            writer.writerow(row)

    print(f"results written to {output_path}")
