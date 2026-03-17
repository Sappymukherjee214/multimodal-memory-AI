import os
import requests
import pandas as pd

def setup_demo_research_data():
    """
    Sets up a small local dataset for immediate multimodal research validation.
    Downloads 3 sample images and creates a corresponding metadata.csv.
    """
    IMG_DIR = "data/raw/images"
    CSV_PATH = "data/raw/metadata.csv"
    os.makedirs(IMG_DIR, exist_ok=True)
    
    # Sample images from public domain (Unsplash/Wiki)
    samples = [
        {"url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb", "name": "nature.jpg", "caption": "A beautiful landscape with mountains and water."},
        {"url": "https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d", "name": "work.jpg", "caption": "A person working on a laptop at a wooden desk."},
        {"url": "https://images.unsplash.com/photo-1517841905240-472988babdf9", "name": "dog.jpg", "caption": "A cute dog sitting on the grass."}
    ]
    
    metadata = []
    
    print("Setting up demo research dataset...")
    for item in samples:
        path = os.path.join(IMG_DIR, item['name'])
        if not os.path.exists(path):
            print(f"Downloading {item['name']}...")
            try:
                response = requests.get(item['url'], stream=True, timeout=10)
                if response.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(response.content)
            except Exception as e:
                print(f"Failed to download {item['name']}: {e}")
        
        metadata.append({
            "image_path": item['name'],
            "caption": item['caption'],
            "timestamp": "2026-03-16 12:00:00"
        })
        
    df = pd.DataFrame(metadata)
    df.to_csv(CSV_PATH, index=False)
    print(f"Demo dataset ready! {len(metadata)} images indexed in {CSV_PATH}")

if __name__ == "__main__":
    setup_demo_research_data()
