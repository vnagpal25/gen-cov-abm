import gzip
import json

gz_file = "dataset.json.gz"
try:
    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        decompressed_data = json.load(f)
    print("\nDecompressed JSON data:")
    print(f"Deompressing json string of length: {len(json.dumps(decompressed_data))}")
except Exception as e:
    print(f"Error decompressing JSON data: {e}")
