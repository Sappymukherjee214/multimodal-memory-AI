from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os
import pandas as pd

class DatasetBootstrapper:
    """
    Research utility to auto-annotate raw multimodal data.
    Uses foundation models (BLIP) to generate initial semantic descriptions,
    enabling self-supervised training on unorganized personal data.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    def generate_captions(self, image_dir, output_csv):
        """
        Iterates through images and generates descriptive captions.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png')
        results = []
        
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
        print(f"Bootstrapping captions for {len(images)} images...")
        
        for img_name in images:
            img_path = os.path.join(image_dir, img_name)
            raw_image = Image.open(img_path).convert('RGB')
            
            # Generate caption
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            results.append({
                "image_path": img_name,
                "caption": caption
            })
            
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Bootstrapping complete. Metadata saved to {output_csv}")

if __name__ == "__main__":
    # This feature enables 'Cold Start' research on unlabelled data
    print("Dataset Bootstrapping module (Self-Supervision) ready.")
