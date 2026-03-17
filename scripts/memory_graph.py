import networkx as nx
import numpy as np
from datetime import datetime

class EpisodicMemoryGraph:
    """
    Research-grade Hierarchical Memory Graph.
    Connects isolated memories into a semantic web based on:
    1. Temporal Proximity (events that happened together)
    2. Semantic Similarity (conceptually related memories)
    3. Spatial Clustering (locations)
    """
    def __init__(self):
        self.graph = nx.Graph()

    def add_memory_node(self, memory_id, metadata, embedding):
        """Adds a memory as a node in the graph with attributes."""
        self.graph.add_node(
            memory_id, 
            timestamp=metadata.get('timestamp', 0),
            caption=metadata.get('caption', ''),
            embedding=embedding
        )
        self._auto_connect_nodes(memory_id)

    def _auto_connect_nodes(self, new_id):
        """Automatically builds edges based on heuristics."""
        new_node = self.graph.nodes[new_id]
        
        for other_id in self.graph.nodes:
            if other_id == new_id: continue
            
            other_node = self.graph.nodes[other_id]
            
            # Heuristic 1: Temporal Context (Within 1 hour)
            time_diff = abs(new_node['timestamp'] - other_node['timestamp'])
            if time_diff < 3600:
                self.graph.add_edge(new_id, other_id, weight=1.5, type='temporal')
            
            # Heuristic 2: Semantic Link
            sim = np.dot(new_node['embedding'], other_node['embedding'])
            if sim > 0.85:
                self.graph.add_edge(new_id, other_id, weight=sim, type='semantic')

    def get_contextual_subgraph(self, memory_id, depth=1):
        """Retrieves nodes 'connected' to the target memory for RAG context expansion."""
        if memory_id not in self.graph:
            return []
            
        # Extract ego network
        subgraph_nodes = nx.ego_graph(self.graph, memory_id, radius=depth).nodes
        return [self.graph.nodes[node] for node in subgraph_nodes]

    def find_memory_clusters(self):
        """Identifies high-level 'events' using community detection."""
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(self.graph)
        return communities

if __name__ == "__main__":
    print("Hierarchical Episodic Memory Graph module ready for personal AI research.")
