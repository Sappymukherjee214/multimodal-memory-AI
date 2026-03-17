import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

class TemporalDecayModule(nn.Module):
    """
    Implements an exponential temporal decay function for memory retrieval.
    Prioritizes recent memories by attenuating similarity scores of older entries.
    Formulation: score_new = score_old * exp(-lambda * delta_t)
    """
    def __init__(self, decay_rate=0.01):
        super(TemporalDecayModule, self).__init__()
        self.decay_rate = decay_rate # Lambda parameter

    def forward(self, scores, timestamps):
        """
        Args:
            scores (Tensor): [Batch, K] - Cosine similarity scores.
            timestamps (list): List of Unix timestamps for the retrieved results.
        Returns:
            weighted_scores (Tensor): Scores adjusted by temporal decay.
        """
        now = datetime.now().timestamp()
        
        # Calculate delta_t in days (or hours, depending on decay_rate scale)
        delta_ts = torch.tensor([now - ts for ts in timestamps]).to(scores.device)
        # Normalize delta_t to days for a more interpretable decay
        delta_days = delta_ts / (24 * 3600)
        
        decay_factors = torch.exp(-self.decay_rate * delta_days)
        
        # Apply decay to scores
        return scores * decay_factors

class MetadataEncoder(nn.Module):
    """
    Encodes contextual metadata (e.g., GPS coordinates, daytime) into 
    secondary embedding weights to refine retrieval rankings.
    """
    def __init__(self, metadata_dim=4, embedding_dim=512):
        super(MetadataEncoder, self).__init__()
        # metadata_dim: [lat, lon, hour_norm, day_norm]
        self.fc = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a weight/bias for the retrieval score
        )

    def forward(self, metadata_vectors):
        """
        Processes metadata vectors into refinement scalars.
        """
        return self.fc(metadata_vectors)

if __name__ == "__main__":
    print("Temporal Decay and Metadata Encoder modules initialized.")
