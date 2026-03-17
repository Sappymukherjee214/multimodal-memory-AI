import os
import pandas as pd
from datetime import datetime

def generate_initial_metadata(image_dir, output_csv):
    """
    Scans an image directory and creates a starter CSV for multimodal training.
    Includes metadata like filename, path, and file creation date.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    data = []
    
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} not found. Creating a simulation guide.")
        return

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(valid_extensions):
            file_path = os.path.join(image_dir, filename)
            # Basic metadata extraction
            ctime = os.path.getctime(file_path)
            dt_object = datetime.fromtimestamp(ctime)
            
            data.append({
                "image_path": filename,
                "caption": "Starter caption: Describe this image.", # Placeholder for manual/auto labeling
                "timestamp": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
                "file_size": os.path.getsize(file_path)
            })
            
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated metadata for {len(data)} images at {output_csv}")

if __name__ == "__main__":
    # Example relative path
    I_DIR = "data/raw/images"
    O_CSV = "data/raw/metadata.csv"
    
    if not os.path.exists(I_DIR):
        os.makedirs(I_DIR, exist_ok=True)
        print(f"Created {I_DIR}. Place your personal images here.")
        
    generate_initial_metadata(I_DIR, O_CSV)
