import os
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings

class VectorMemoryIndex:
    """
    Research-grade Vector Storage and Indexing system.
    Supports FAISS for high-speed similarity search and ChromaDB for 
    persistent metadata management and easy integration with RAG.
    """
    def __init__(self, embedding_dim=512, use_chroma=True, persist_directory="data/processed/vector_store"):
        self.embedding_dim = embedding_dim
        self.use_chroma = use_chroma
        self.persist_directory = persist_directory
        
        # 1. Initialize FAISS Index (HNSW for scaling to millions of entries)
        # HNSW provides logarithmic search time and high recall.
        self.faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32) # 32 neighbors per node
        
        # 2. Initialize ChromaDB for persistent storage and metadata
        if self.use_chroma:
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.chroma_client.get_or_create_collection(name="multimodal_memory")

    def add_memories(self, embeddings, metadata_list, ids_list):
        """
        Args:
            embeddings (ndarray): [N, Dim] - Feature vectors.
            metadata_list (list): List of dicts containing timestamps, paths, etc.
            ids_list (list): List of unique string identifiers.
        """
        # Add to FAISS (In-memory acceleration)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings).astype('float32')
        self.faiss_index.add(embeddings)
        
        # Add to Chroma (Persistence and Metadata search)
        if self.use_chroma:
            self.collection.add(
                embeddings=embeddings.tolist(),
                metadatas=metadata_list,
                ids=ids_list
            )
        print(f"Index updated: Added {len(ids_list)} memories.")

    def search(self, query_embedding, k=5):
        """
        Performs sub-millisecond similarity search.
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding).astype('float32')
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # FAISS search for raw indices and scores
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # ChromaDB search if persistence is required
        if self.use_chroma:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
            return results
        
        return {"distances": distances, "indices": indices}

    def save_faiss(self, path="data/processed/faiss_index.bin"):
        faiss.write_index(self.faiss_index, path)
        print(f"FAISS index saved to {path}")

    def load_faiss(self, path="data/processed/faiss_index.bin"):
        self.faiss_index = faiss.read_index(path)
        print("FAISS index loaded.")

if __name__ == "__main__":
    print("Vector Memory Index (FAISS + ChromaDB) initialized.")
