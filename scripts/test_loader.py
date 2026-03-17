import os
import pandas as pd
import sys
sys.path.append(os.getcwd())

from data.loaders.multimodal_dataset import MultimodalDataset

DATA_PATH = os.environ.get("DATASET_PATH", "data/raw")
IMG_DIR = os.path.join(DATA_PATH)
CSV_FILE = os.path.join(DATA_PATH, "metadata.csv")

try:
    if os.path.exists(CSV_FILE):
        dataset = MultimodalDataset(CSV_FILE, IMG_DIR)
        print(f"Dataset length: {len(dataset)}")
        print(f"First item keys: {dataset[0].keys()}")
    else:
        print(f"Dataset file not found at {CSV_FILE}. Please provide a valid dataset.")
except Exception as e:
    import traceback
    traceback.print_exc()
