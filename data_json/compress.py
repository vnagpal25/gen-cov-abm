import gzip
import json


file_name = "tree.json"
with open(file_name, "r") as f:
    data = json.load(f)

print(f"Compressing json string of length: {len(json.dumps(data))}")
output_file = f"{file_name}.gz"
try:
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(data, f)
except Exception as e:
    print(f"Error compresssing JSON data: {e}")
