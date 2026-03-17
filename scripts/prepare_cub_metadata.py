import os
import pandas as pd
from tqdm import tqdm

def generate_cub_metadata(cub_root, text_root, output_csv):
    """
    Parses CUB-200-2011 and its associated captions to create a unified metadata.csv.
    """
    images_txt = os.path.join(cub_root, "images.txt")
    
    if not os.path.exists(images_txt):
        print(f"Error: {images_txt} not found.")
        return

    # Load image mapping
    print("Reading image mappings...")
    image_df = pd.read_csv(images_txt, sep=" ", names=["id", "rel_path"])
    
    data = []
    
    print("Processing images and captions...")
    for idx, row in tqdm(image_df.iterrows(), total=len(image_df)):
        img_rel_path = row["rel_path"]
        img_id_str = os.path.splitext(img_rel_path)[0] # e.g. 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18
        
        # Caption file path
        caption_file = os.path.join(text_root, img_id_str + ".txt")
        
        if os.path.exists(caption_file):
            with open(caption_file, "r") as f:
                captions = [line.strip() for line in f.readlines() if line.strip()]
            
            # We take all 10 captions as separate entries (like Flickr does)
            for cap in captions:
                data.append({
                    "image_path": os.path.join("CUB_200_2011", "images", img_rel_path),
                    "caption": cap,
                    "dataset": "CUB-200-2011"
                })
        else:
            # Fallback if no caption found
            pass

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\nMetadata generated with {len(df)} caption-image pairs.")
    print(f"Saved to: {output_csv}")

if __name__ == "__main__":
    CUB_ROOT = r"c:\Users\DELL\OneDrive\Desktop\Multimodal Memory AI\data\raw\CUB_200_2011"
    TEXT_ROOT = r"C:\Users\DELL\.cache\kagglehub\datasets\wenewone\cub2002011\versions\7\cvpr2016_cub\text_c10"
    OUTPUT_CSV = r"c:\Users\DELL\OneDrive\Desktop\Multimodal Memory AI\data\raw\metadata.csv"
    
    generate_cub_metadata(CUB_ROOT, TEXT_ROOT, OUTPUT_CSV)
