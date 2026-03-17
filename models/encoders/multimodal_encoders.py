import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoConfig, ClapModel, ClapProcessor
import torch.nn.functional as F

class VisionEncoder(nn.Module):
    """
    Encoder for visual features using Vision Transformer (ViT) or ResNet backbones.
    Includes a projection head to map features to the shared embedding space.
    """
    def __init__(self, model_name='vit_b_16', pretrained=True, embedding_dim=512):
        super(VisionEncoder, self).__init__()
        if model_name == 'vit_b_16':
            self.model = models.vit_b_16(weights='DEFAULT' if pretrained else None)
            self.feature_dim = self.model.heads.head.in_features
            self.model.heads = nn.Identity() # Remove classification head
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported vision model: {model_name}")

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x, return_sequence=False):
        if hasattr(self.model, 'forward_features'): # For modern ViT implementations if used
             features = self.model.forward_features(x)
        else:
            # Fallback for standard torchvision ViT
            if isinstance(self.model, models.VisionTransformer):
                # We need to extract tokens before the final pooling
                x = self.model._process_input(x)
                n = x.shape[0]
                batch_class_token = self.model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = self.model.encoder(x)
                features = x # [Batch, SeqLen, Dim]
            else:
                features = self.model(x)

        if return_sequence:
            # Project patch features to embedding space
            seq_features = self.projection(features)
            return seq_features # Return full patch sequence for cross-attention
            
        # Standard pooled output
        if features.dim() == 3:
            pooled = features[:, 0] # Use [CLS] token
        else:
            pooled = features
            
        embeddings = self.projection(pooled)
        return nn.functional.normalize(embeddings, p=2, dim=1)

class TextEncoder(nn.Module):
    """
    Encoder for textual features using Transformer-based models (e.g., BERT, RoBERTa).
    Includes a projection head to map features to the shared embedding space.
    """
    def __init__(self, model_name='bert-base-uncased', embedding_dim=512):
        super(TextEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, input_ids, attention_mask, return_sequence=False):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        if return_sequence:
            # Apply projection to the sequence features to match vision space
            seq_features = self.projection(last_hidden_state)
            return seq_features # Return token sequence for cross-attention

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        features = sum_embeddings / sum_mask
        
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)

class AudioEncoder(nn.Module):
    """
    Encoder for audio features using CLAP (Contrastive Language-Audio Pretraining).
    """
    def __init__(self, model_name='laion/clap-htsat-fused', embedding_dim=512):
        super(AudioEncoder, self).__init__()
        self.model = ClapModel.from_pretrained(model_name)
        self.feature_dim = self.model.config.projection_dim
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, input_features, return_sequence=False):
        # CLAP expects input_features from a processor
        outputs = self.model.get_audio_features(input_features)
        
        if return_sequence:
            # Note: Standard CLAP doesn't easily return patch-level audio tokens 
            # without deeper model access. For now, we return the pooled feature.
            return outputs.unsqueeze(1) 

        embeddings = self.projection(outputs)
        return F.normalize(embeddings, p=2, dim=1)

class ContextualEncoder(nn.Module):
    """
    Wraps a base encoder (Vision/Text/Audio) and integrates metadata signals
    (Time, Location) into the final embedding.
    """
    def __init__(self, base_encoder, metadata_dim=4, embedding_dim=512):
        super(ContextualEncoder, self).__init__()
        self.base_encoder = base_encoder
        
        # Meta-Projection Head
        self.meta_proj = nn.Sequential(
            nn.Linear(embedding_dim + 128, embedding_dim), # 128 is metadata hidden dim
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Simple metadata MLP (lat, lon, time_norm, day_norm)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x, metadata_vec=None, **kwargs):
        # 1. Base modality embedding
        base_emb = self.base_encoder(x, **kwargs)
        
        if metadata_vec is None:
            return base_emb
            
        # 2. Metadata signal
        meta_signal = self.metadata_mlp(metadata_vec)
        
        # 3. Contextual Fusion
        combined = torch.cat([base_emb, meta_signal], dim=-1)
        fused_emb = self.meta_proj(combined)
        
        return F.normalize(fused_emb, p=2, dim=1)

if __name__ == "__main__":
    print("Multi-modal encoders defined with standard backbones and projection heads.")
