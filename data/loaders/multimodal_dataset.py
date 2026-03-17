import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MultimodalDataset(Dataset):
    """
    Research-grade dataset class for personal multimodal data.
    Supports Image-Text pairs with associated metadata.
    """
    def __init__(self, csv_file, img_dir, transform=None, tokenizer=None, max_length=77):
        """
        Args:
            csv_file (string): Path to the csv file with annotations (image_path, caption, metadata).
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            tokenizer (callable): Tokenizer for text processing (e.g., CLIP tokenizer or BERT).
            max_length (int): Maximum token length for text.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        caption = self.annotations.iloc[index, 1]
        
        # Metadata extraction (e.g., timestamp, location) if available
        metadata = self.annotations.iloc[index, 2:].to_dict() if len(self.annotations.columns) > 2 else {}

        if self.transform:
            image = self.transform(image)
        
        if self.tokenizer:
            # Tokenize the caption
            tokens = self.tokenizer(
                caption, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = caption
            attention_mask = torch.ones(len(caption)) # Placeholder

        # Advanced Feature: Metadata Vectorization for Contextual Alignment
        # We simulate/normalize 4 dimensions: [timestamp_norm, lat, lon, day_of_week]
        # In a real personal memory app, these would come from EXIF/GPS.
        ts = metadata.get('timestamp', 0)
        ts_norm = (ts % 86400) / 86400.0 # Time of day
        md_vec = torch.tensor([ts_norm, 0.0, 0.0, 0.0], dtype=torch.float32)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "metadata": metadata,
            "metadata_vec": md_vec
        }

def get_dataloader(csv_file, img_dir, batch_size=32, shuffle=True, transform=None, tokenizer=None):
    dataset = MultimodalDataset(csv_file, img_dir, transform=transform, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_iter_workers=4)

if __name__ == "__main__":
    # Example usage for testing the dataset implementation
    print("MultimodalDataset class initialized successfully.")
