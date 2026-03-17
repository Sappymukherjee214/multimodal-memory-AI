# Technical Documentation: Multimodal Memory AI (MMAI)

This document provides a detailed technical reference for the MMAI framework, covering the architecture, module specifications, and mathematical foundations.

---

## 🏗️ 1. System Architecture Overview

The MMAI framework is designed as a modular, two-stage retrieval system specialized for personal episodic data. It separates high-speed coarse filtering from high-fidelity deep interaction.

### 1.1 Core Components
*   **Encoder Layer:** Projects raw pixels, text, and audio into a unified 512-D latent space.
*   **Vector Engine:** Manages persistent storage and sub-millisecond similarity search using FAISS (IndexHNSWFlat).
*   **Associative Graph:** A NetworkX-based episodic web that links memories based on temporal and semantic proximity.
*   **Fusion Transformer:** A secondary re-ranking layer implementing patch-to-token cross-modal attention.

---

## 🔌 2. API Reference

### 2.1 `MultimodalMemoryCore`
Located in `scripts/memory_core.py`. This is the central controller for the framework.

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `index_image` | `image_path`, `metadata` | `memory_id` | Encodes and indexes an image with metadata into vector & graph stores. |
| `search_by_text` | `query_text`, `k`, `do_rerank` | `List[Dict]` | Performs stage-1 search or stage-2 re-ranked search. |
| `search_with_associations` | `query_text`, `k` | `Dict` | Performs Graph-Augmented Retrieval (GAR) to find related episodic context. |
| `explain_retrieval` | `query_text`, `image_path` | `ndarray` | Generates a 14x14 attention heatmap for interpretability. |

### 2.2 `VisionEncoder` & `TextEncoder`
Located in `models/encoders/multimodal_encoders.py`.

*   **VisionEncoder:** Uses ViT-B/16. Returns global [CLS] embedding (pooled) or patch sequence (unpooled).
*   **TextEncoder:** Uses BERT-Base-Uncased. Returns mean-pooled embedding or token sequence.
*   **ContextualEncoder Wrapper:** Combines base modality embeddings with a 4D metadata vector (timestamp, location).

### 2.3 `MultimodalFusionTransformer`
Located in `models/fusion/cross_modal_fusion.py`.

*   **Input:** Image patch sequence $[B, 197, 512]$ and Text token sequence $[B, 77, 512]$.
*   **Logic:** Iterative cross-attention where text "queries" image patches to compute a refined interaction score.

---

## 📉 3. Mathematical Foundations

### 3.1 Hard Negative Mining (HNM)
We use a momentum-updated queue $\mathbf{Q}$ to maintain a diverse set of negative samples $\mathcal{N}$ during contrastive training. For a positive pair $(v_i, t_i)$, the loss is defined as:

$$ \mathcal{L} = -\log \frac{\exp(v_i \cdot t_i / \tau)}{\exp(v_i \cdot t_i / \tau) + \sum_{k=1}^{|\mathbf{Q}|} \exp(v_i \cdot \mathbf{Q}_k / \tau)} $$

Where $\tau$ is the temperature parameter (default 0.07).

### 3.2 Episodic Decay
Retrieval scores $S$ are attenuated over time $t$ using an exponential decay function in `TemporalDecayModule`:
$$ S_{refined} = S_{raw} \cdot e^{-\lambda \Delta t} $$
Where $\Delta t$ is the normalized time difference in days.

---

## 🛠️ 4. Maintenance & Extensions

### Adding a New Modality
1.  Define a new encoder class in `models/encoders/multimodal_encoders.py`.
2.  Ensure it outputs a 512-D vector (or use a projection layer).
3.  Update `MultimodalMemoryCore` to include an indexing method for the new modality.

### Custom Dataset Integration
The `MultimodalDataset` class in `data/loaders/multimodal_dataset.py` is generic. To use a new dataset, prepare a CSV with:
*   `image_path`: Relative to the root image directory.
*   `caption`: The natural language description.
*   `timestamp`: Unix timestamp (optional).

---

## 📊 5. Evaluation Metrics
The framework provides an automated suite in `scripts/evaluate.py`:
*   **Recall@K (I2T/T2I):** Standard accuracy metric for cross-modal retrieval.
*   **Saliency Mean-IoU:** (External) For measuring alignment between human gaze and model attention.

---

*Documentation compiled by MMAI Research Team.*
