import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """
    Research-grade Cross-Modal Attention module.
    Allows features from one modality (e.g., Text) to attend to features
    from another (e.g., Image patches) to capture fine-grained semantic relations.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query_modal, key_value_modal):
        """
        Args:
            query_modal (Tensor): [Batch, SeqLen_Q, Dimension]
            key_value_modal (Tensor): [Batch, SeqLen_KV, Dimension]
        Returns:
            fused_features (Tensor): Features from query_modal enriched with info from key_value_modal.
        """
        # Cross-Attention: Query attends to Key/Value
        attn_output, _ = self.multihead_attn(query_modal, key_value_modal, key_value_modal)
        x = self.norm(query_modal + attn_output)
        
        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class MultimodalFusionTransformer(nn.Module):
    """
    Late-fusion transformer that integrates multi-modal features after extraction.
    """
    def __init__(self, embed_dim=512, depth=2):
        super(MultimodalFusionTransformer, self).__init__()
        self.layers = nn.ModuleList([
            CrossModalAttention(embed_dim) for _ in range(depth)
        ])

    def forward(self, image_features, text_features):
        # image_features: [Batch, N_patches, Dim]
        # text_features: [Batch, Seq_len, Dim]
        
        # Iterative cross-attention
        for layer in self.layers:
            # Text attends to Image
            text_features = layer(text_features, image_features)
            # Image attends to Text
            image_features = layer(image_features, text_features)
            
        return image_features, text_features

if __name__ == "__main__":
    print("Cross-Modal Attention modules initialized.")
