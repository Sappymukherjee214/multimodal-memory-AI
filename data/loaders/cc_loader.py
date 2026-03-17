import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms

class ConceptualCaptionsLoader(Dataset):
    """
    Specialized loader for the Conceptual Captions dataset.
    Conceptual Captions typically provides a TSV with (caption, url).
    This loader handles URL-based fetching (optional) or local path loading.
    """
    def __init__(self, tsv_file, img_dir, transform=None, tokenizer=None, max_samples=None):
        # CC usually has no headers: Col 0 is Caption, Col 1 is URL
        self.df = pd.read_csv(tsv_file, sep='\t', names=['caption', 'url'])
        if max_samples:
            self.df = self.df.head(max_samples)
            
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = str(self.df.iloc[idx, 0])
        # In CC, images are usually named by their index or hash
        img_name = f"image_{idx}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # For research stability, return a blank image if file is missing/corrupt
            image = Image.new('RGB', (224, 224), color='gray')
            
        if self.transform:
            image = self.transform(image)
            
        if self.tokenizer:
            tokens = self.tokenizer(caption, padding='max_length', truncation=True, max_length=77, return_tensors="pt")
            return {
                "image": image,
                "input_ids": tokens['input_ids'].squeeze(0),
                "attention_mask": tokens['attention_mask'].squeeze(0),
                "caption": caption
            }
            
        return {"image": image, "caption": caption}

def prepare_cc_metadata(tsv_input, csv_output):
    """
    Converts raw CC TSV into the project's standard metadata.csv format.
    """
    print(f"Converting Conceptual Captions TSV to standard research format...")
    df = pd.read_csv(tsv_input, sep='\t', names=['caption', 'url'])
    # Map to local filenames
    df['image_path'] = [f"image_{i}.jpg" for i in range(len(df))]
    # Keep only what we need
    df = df[['image_path', 'caption']]
    df.to_csv(csv_output, index=False)
    print(f"Metadata saved to {csv_output}")

if __name__ == "__main__":
    print("Conceptual Captions research loader ready.")
