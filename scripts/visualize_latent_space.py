import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import os

def visualize_latent_space(image_embeddings, text_embeddings, labels=None, output_path="reports/figures/latent_space_umap.png"):
    """
    Project high-dimensional multimodal embeddings into 2D space using UMAP.
    Verifies if image/text pairs are correctly clustered together.
    """
    print(f"Visualizing latent space for {len(image_embeddings)} items...")
    
    # Combine embeddings for joint projection
    all_embeddings = np.concatenate([image_embeddings, text_embeddings], axis=0)
    modality_labels = ['Image'] * len(image_embeddings) + ['Text'] * len(text_embeddings)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding_2d[:, 0], 
        y=embedding_2d[:, 1], 
        hue=modality_labels, 
        style=modality_labels,
        alpha=0.6,
        palette='Set1'
    )
    
    if labels is not None:
        # Draw connections between matched pairs for visual verification
        for i in range(len(image_embeddings)):
            plt.plot(
                [embedding_2d[i, 0], embedding_2d[i + len(image_embeddings), 0]],
                [embedding_2d[i, 1], embedding_2d[i + len(image_embeddings), 1]],
                'gray', alpha=0.1, linewidth=0.5
            )

    plt.title("UMAP Projection: Multimodal Alignment Verification")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Latent space plot saved to {output_path}")

def plot_retrieval_grid(query_text, retrieved_images, scores, output_path="reports/figures/retrieval_verification.png"):
    """
    Generate a grid of retrieved images for a specific query for qualitative research validation.
    """
    n = len(retrieved_images)
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    fig.suptitle(f"Query: '{query_text}'", fontsize=16)
    
    for i in range(n):
        axes[i].imshow(retrieved_images[i])
        axes[i].set_title(f"Score: {scores[i]:.4f}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Retrieval verification grid saved to {output_path}")

if __name__ == "__main__":
    print("Advanced Visualization tools for latent space analysis ready.")
