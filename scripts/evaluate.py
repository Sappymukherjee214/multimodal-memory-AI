import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Computes Recall@K for image-to-text and text-to-image retrieval.
    Args:
        similarity_matrix (ndarray): [Queries, Candidates] similarity scores.
        k_values (list): Values of K to calculate recall for.
    Returns:
        dict: Recall scores for each K.
    """
    num_queries = similarity_matrix.shape[0]
    ranks = np.argsort(similarity_matrix, axis=1)[:, ::-1] # Sort descending
    
    # Ground truth for query i is index i
    ground_truth = np.arange(num_queries).reshape(-1, 1)
    
    results = {}
    for k in k_values:
        # Check if ground truth index is within the top K predicted indices
        hits = np.any(ranks[:, :k] == ground_truth, axis=1)
        results[f"Recall@{k}"] = np.sum(hits) / num_queries
        
    return results

def evaluate_retrieval(vision_encoder, text_encoder, dataloader, device):
    """
    Runs evaluation on the dataset to measure cross-modal alignment quality.
    """
    vision_encoder.eval()
    text_encoder.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            img_emb = vision_encoder(images)
            txt_emb = text_encoder(input_ids, attention_mask)
            
            all_image_embeddings.append(img_emb.cpu().numpy())
            all_text_embeddings.append(txt_emb.cpu().numpy())
            
    # Concatenate all batches
    img_embs = np.concatenate(all_image_embeddings, axis=0)
    txt_embs = np.concatenate(all_text_embeddings, axis=0)
    
    # Similarity matrix
    similarity = cosine_similarity(img_embs, txt_embs)
    
    # Image-to-Text Recall
    i2t_recall = calculate_recall_at_k(similarity)
    
    # Text-to-Image Recall
    t2i_recall = calculate_recall_at_k(similarity.T)
    
    return {
        "I2T_Recall": i2t_recall,
        "T2I_Recall": t2i_recall
    }

if __name__ == "__main__":
    print("Multimodal evaluation suite (Recall@K) ready.")
