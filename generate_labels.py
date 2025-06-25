import os
import csv

dataset_path = "data/images"
output_csv = "data/labels.csv"

data = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = os.path.basename(root)
            rel_path = os.path.relpath(os.path.join(root, file), dataset_path)
            data.append((rel_path.replace("\\", "/"), label))

os.makedirs("data", exist_ok=True)

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "class"])
    writer.writerows(data)

print(f"✅ labels.csv generated with {len(data)} entries.")
