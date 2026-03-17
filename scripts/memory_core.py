import torch
import os
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
import numpy as np

# Internal imports
from models.encoders.multimodal_encoders import VisionEncoder, TextEncoder
from models.alignment.contextual_alignment import TemporalDecayModule
from models.fusion.cross_modal_fusion import MultimodalFusionTransformer
from scripts.vector_storage import VectorMemoryIndex

class MultimodalMemoryCore:
    """
    The 'Central Nervous System' of the personal memory retrieval agent.
    Advanced version with Two-Stage Retrieval and Explainability.
    """
    def __init__(self, 
                 vision_model_path=None, 
                 text_model_path=None, 
                 decay_rate=0.01,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.embedding_dim = 512
        
        # 1. Initialize Encoders
        print(f"Initializing Memory Core on {device}...")
        self.vision_encoder = VisionEncoder(embedding_dim=self.embedding_dim).to(device)
        self.text_encoder = TextEncoder(embedding_dim=self.embedding_dim).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # 2. Advanced Alignment & Fusion
        # Re-ranker: Deep fusion via cross-modal attention
        # We project the encoders' dimension to a common latent dimension for fusion
        self.reranker = MultimodalFusionTransformer(embed_dim=768).to(device) # ViT-Base dim is 768
        self.temporal_decay = TemporalDecayModule(decay_rate=decay_rate).to(device)

        # (Optional) Load Fine-tuned Weights
        if vision_model_path:
            self.vision_encoder.load_state_dict(torch.load(vision_model_path, map_location=device))
        if text_model_path:
            self.text_encoder.load_state_dict(torch.load(text_model_path, map_location=device))
            
        self.vision_encoder.eval()
        self.text_encoder.eval()
        self.reranker.eval()

        # 3. Initialize Vector Storage
        self.storage = VectorMemoryIndex(embedding_dim=self.embedding_dim)

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 4. Initialize Episodic Graph
        from scripts.memory_graph import EpisodicMemoryGraph
        self.memory_graph = EpisodicMemoryGraph()

    def index_image(self, image_path, metadata):
        """Indexes an image into both the vector store and the associative graph."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.img_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.vision_encoder(image_tensor).cpu().numpy()
            
        memory_id = f"img_{np.random.randint(1000000)}"
        
        # Ensure metadata
        if 'timestamp' not in metadata:
            import time
            metadata['timestamp'] = int(time.time())
        metadata['image_path'] = image_path
            
        # Add to Vector Core
        self.storage.add_memories(embedding, [metadata], [memory_id])
        
        # Add to Associative Graph
        self.memory_graph.add_memory_node(memory_id, metadata, embedding[0])
        
        return memory_id

    def rerank_candidates(self, query_text, candidates, k=3):
        """
        Stage 2: Semantic Re-ranking using Cross-Modal Attention.
        Takes candidate memories and performs fine-grained interacton search.
        """
        print(f"Re-ranking top {len(candidates)} candidates...")
        tokens = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        scores = []
        with torch.no_grad():
            # Get text sequence features
            text_seq = self.text_encoder(tokens['input_ids'], tokens['attention_mask'], return_sequence=True)
            
            for candidate in candidates:
                img_path = candidate.get('image_path')
                if not img_path or not os.path.exists(img_path):
                    scores.append(-1.0)
                    continue
                
                # Get image patch sequence features
                image = Image.open(img_path).convert("RGB")
                img_tensor = self.img_transform(image).unsqueeze(0).to(self.device)
                img_seq = self.vision_encoder(img_tensor, return_sequence=True)
                
                # Fusion
                fused_img, fused_txt = self.reranker(img_seq, text_seq)
                
                # Fine-grained score: similarity between fused CLS tokens
                score = torch.cosine_similarity(fused_img[:, 0], fused_txt[:, 0]).item()
                scores.append(score)
        
        # Sort candidates by re-ranker score
        ranked_indices = np.argsort(scores)[::-1]
        return [candidates[i] for i in ranked_indices[:k]]

    def explain_retrieval(self, query_text, image_path):
        """
        Explainability Module: Vision-Language Saliency.
        Generates a heatmap showing which parts of the image match the query.
        """
        print(f"Generating semantic saliency map for: '{query_text}'")
        tokens = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.img_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            text_seq = self.text_encoder(tokens['input_ids'], tokens['attention_mask'], return_sequence=True)
            img_seq = self.vision_encoder(img_tensor, return_sequence=True)
            
            # Cross-attention heatmaps
            # For simplicity, we use the dot product of averaged text tokens with each patch
            text_avg = text_seq.mean(dim=1, keepdim=True) # [1, 1, Dim]
            # Patch sequence excluding CLS token
            patches = img_seq[:, 1:, :] # [1, 196, Dim] for 224x224 ViT
            
            saliency = torch.matmul(patches, text_avg.transpose(1, 2)).squeeze() # [196]
            saliency = saliency.cpu().numpy().reshape(14, 14) # 196 patches = 14x14 grid
            
        return saliency

    def search_by_text(self, query_text, k=10, apply_decay=True, do_rerank=True):
        """Search with Optional Two-Stage Re-ranking."""
        tokens = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.text_encoder(tokens['input_ids'], tokens['attention_mask']).cpu().numpy()
            
        results = self.storage.search(query_embedding, k=k)
        
        # Stage 1: Retrieval
        candidates = results.get('metadatas', [[]])[0]
        
        # Apply Temporal Decay
        if apply_decay and candidates:
            raw_scores = torch.tensor(results['distances'][0]).to(self.device)
            timestamps = [m.get('timestamp', 0) for m in candidates]
            refined_scores = self.temporal_decay(raw_scores, timestamps)
            
        # Stage 2: Re-ranking
        if do_rerank and len(candidates) > 1:
            candidates = self.rerank_candidates(query_text, candidates, k=min(3, len(candidates)))
            
        return candidates

    def search_with_associations(self, query_text, k=3):
        """
        Advanced GAR (Graph-Augmented Retrieval).
        Finds direct matches AND related memories via graph walk.
        """
        # 1. Get primary semantic candidates
        primary_candidates = self.search_by_text(query_text, k=k, do_rerank=True)
        
        # 2. Extract associative context for each candidate
        expanded_memories = []
        seen_paths = {c.get('image_path') for c in primary_candidates}
        
        for cand in primary_candidates:
            # We need the memory_id to look in graph. 
            # In a real system, we'd store ID in metadata.
            # For demo, we search by path similarity in graph if ID is missing
            for node_id, data in self.memory_graph.graph.nodes(data=True):
                if data.get('image_path') == cand.get('image_path'):
                    context = self.memory_graph.get_contextual_subgraph(node_id, depth=1)
                    for ctx_node in context:
                        if ctx_node.get('image_path') not in seen_paths:
                            ctx_node['association_type'] = "Graph-Linked"
                            expanded_memories.append(ctx_node)
                            seen_paths.add(ctx_node.get('image_path'))
        
        return {
            "primary": primary_candidates,
            "associative": expanded_memories[:k]
        }

    def search_by_image(self, image_path, k=5):
        """Perform Image-to-Image retrieval."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.img_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.vision_encoder(image_tensor).cpu().numpy()
            
        results = self.storage.search(query_embedding, k=k)
        return results

if __name__ == "__main__":
    print("Multimodal Memory Core v2.0 (Advanced Research Edition) ready.")
