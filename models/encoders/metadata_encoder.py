import torch
import torch.nn as nn
import numpy as np

class ContextualMetadataEncoder(nn.Module):
    """
    Encodes contextual signals like Time and Location (GPS) into embeddings.
    Used for weighting retrieval based on 'Memory Recency' or 'Place Similarity'.
    """
    def __init__(self, output_dim=512):
        super(ContextualMetadataEncoder, self).__init__()
        
        # Temporal Encoding (e.g., Sinusoidal Time Encoding)
        self.time_proj = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim // 2)
        )
        
        # Spatial Encoding (Lat/Long pairs)
        self.geo_proj = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim // 2)
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, normalized_timestamp, lat_long=None):
        """
        Args:
            normalized_timestamp (Tensor): [Batch, 1] - Scaled time value.
            lat_long (Tensor): [Batch, 2] - GPS Coordinates.
        Returns:
            context_embedding (Tensor): [Batch, output_dim]
        """
        t_emb = self.time_proj(normalized_timestamp)
        
        if lat_long is not None:
            g_emb = self.geo_proj(lat_long)
        else:
            g_emb = torch.zeros_like(t_emb)
            
        combined = torch.cat([t_emb, g_emb], dim=-1)
        return self.final_projection(combined)

class MemoryDecayModule(nn.Module):
    """
    Implements the 'Forgetfulness' curve in semantic memory.
    Items that are older or less contextually relevant have their similarity scores attenuated.
    """
    def __init__(self, decay_rate=0.1):
        super(MemoryDecayModule, self).__init__()
        self.decay_rate = decay_rate

    def forward(self, similarity_scores, delta_time):
        # delta_time: [Batch, Candidates] time difference in days/hours
        decay_factor = torch.exp(-self.decay_rate * delta_time)
        return similarity_scores * decay_factor

if __name__ == "__main__":
    print("Contextual Metadata Encoders and Memory Decay modules initialized.")
