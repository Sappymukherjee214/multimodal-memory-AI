import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Enhanced InfoNCE loss with Hard Negative Mining (HNM).
    Focuses the learning on the most challenging negative samples in the batch.
    """
    def __init__(self, temperature=0.07, use_hard_negative=True, negative_weight=1.5, queue_size=4096):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.use_hard_negative = use_hard_negative
        self.negative_weight = negative_weight
        self.queue_size = queue_size
        
        # Momentum Queue (Global Hard Negatives)
        self.register_buffer("queue", torch.randn(512, queue_size))
        self.queue = F.normalize(self.queue, p=2, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_queue(self, keys):
        """
        Updates the momentum queue with new embeddings. 
        Must be called AFTER the backward pass to avoid in-place modification errors.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the oldest embeddings in the queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.detach().T
        else:
            # Handle wrap-around
            overflow = (ptr + batch_size) - self.queue_size
            self.queue[:, ptr:] = keys.detach().T[:, :batch_size-overflow]
            self.queue[:, :overflow] = keys.detach().T[:, batch_size-overflow:]
            
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, image_embeddings, text_embeddings):
        batch_size = image_embeddings.size(0)
        
        # 1. Intra-batch similarities (Positive Pairs on diagonal)
        logits_internal = torch.matmul(image_embeddings, text_embeddings.transpose(0, 1)) / self.temperature
        
        # 2. Global similarities (Contrast against Queue)
        # We use the current queue state. No modification happens here.
        logits_external = torch.matmul(image_embeddings, self.queue.to(image_embeddings.device)) / self.temperature
        
        # Concatenate for full contrastive perspective
        # logits shape: [Batch, Batch + Queue_Size]
        logits = torch.cat([logits_internal, logits_external], dim=1)
        
        if self.use_hard_negative:
            # Mask for positive pairs
            mask = torch.zeros(batch_size, batch_size + self.queue_size).to(image_embeddings.device)
            for i in range(batch_size):
                mask[i, i] = 1.0
            
            # Weighted loss for hard negatives
            logits = logits * (1 - mask) * self.negative_weight + logits * mask

        # Labels are simply the indices of the positive samples
        labels = torch.arange(batch_size).to(image_embeddings.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss

if __name__ == "__main__":
    print("InfoNCELoss implemented for multimodal alignment.")
    # Simple test
    loss_fn = InfoNCELoss()
    img_emb = torch.randn(4, 512)
    txt_emb = torch.randn(4, 512)
    img_emb = F.normalize(img_emb, p=2, dim=1)
    txt_emb = F.normalize(txt_emb, p=2, dim=1)
    loss = loss_fn(img_emb, txt_emb)
    print(f"Sample Loss Value: {loss.item():.4f}")
