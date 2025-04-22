import os
import json
import csv

def ensure_dir(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def save_json(data, filepath):
    """Save a dictionary to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def write_csv(filepath, rows, header=None):
    """Write rows to a CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
