import torch
import torch.nn.functional as F
import numpy as np
import cv2

class CLIPExplainability:
    """
    Research tool to visualize 'where' the model is looking in an image
    relative to a specific text query (Semantic Saliency).
    """
    def __init__(self, model):
        self.model = model
    
    def get_attention_map(self, vision_encoder, image_tensor, text_embedding):
        """
        Calculates the similarity between the text embedding and 
        spatial features (patch embeddings) of the vision transformer.
        """
        vision_encoder.eval()
        with torch.no_grad():
            # Use the research-grade return_sequence interface
            # features shape: [Batch, SeqLen (1 + N_patches), Dim]
            features = vision_encoder(image_tensor, return_sequence=True)
            
            # Remove [CLS] token if present (it's at index 0)
            if features.shape[1] > 1:
                features = features[:, 1:, :]
            
        # Normalize features
        features = F.normalize(features, p=2, dim=-1) # [Batch, N_patches, Dim]
        
        # Calculate cosine similarity for each patch
        # text_embedding: [Batch, Dim]
        # Similarity: [Batch, N_patches]
        similarity = torch.matmul(features, text_embedding.unsqueeze(-1)).squeeze(-1)
        
        # Reshape to 2D grid
        batch_size, n_patches = similarity.shape
        grid_size = int(np.sqrt(n_patches))
        heatmap = similarity.view(batch_size, grid_size, grid_size)
        
        return heatmap

def overlay_heatmap(image_np, heatmap):
    """
    Overlays calculated attention heatmap onto the original image.
    """
    heatmap = heatmap.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.uint8(255 * heatmap)
    
    # Resize and blur for smooth visualization
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlayed = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return overlayed

if __name__ == "__main__":
    print("Explainability tools (Semantic Saliency Maps) initialized.")
