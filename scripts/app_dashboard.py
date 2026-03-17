import gradio as gr
import torch
import os
import numpy as np
from PIL import Image
from scripts.memory_core import MultimodalMemoryCore
from scripts.memory_rag import PersonalMemoryRAG
from scripts.explain_retrieval import overlay_heatmap, CLIPExplainability

# Resolve OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Initialize System
print("Loading Multimodal Memory System for UI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use default pre-trained for demo if no fine-tuned weights exist
memory_core = MultimodalMemoryCore(device=device)
memory_rag = PersonalMemoryRAG(memory_core)
explainer = CLIPExplainability(None) # Model internal access is handled in explain_retrieval

def multimodal_search(query, search_mode, k=3):
    """
    Unified search function for the Gradio UI.
    """
    if search_mode == "Text-to-Image":
        results = memory_core.search_by_text(query, k=k, do_rerank=True)
    else:
        # For simplicity, if query is an image path in image-to-image mode
        results = memory_core.search_by_image(query, k=k)
        # Results from FAISS/Chroma search_by_image have different format
        results = results.get('metadatas', [[]])[0]

    # Process results for UI display
    output_html = ""
    images = []
    captions = []
    
    for i, res in enumerate(results):
        img_path = res.get('image_path')
        caption = res.get('caption', 'No caption')
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            captions.append(f"Rank {i+1}: {caption}")
            
    return images, "\n".join(captions)

def generate_rag_answer(query):
    print(f"RAG Request: {query}")
    rag_bundle = memory_rag.query_with_memory(query, k=3)
    return rag_bundle['response'], rag_bundle['prompt']

def explain_result(query, image):
    # This requires the image to be one of the indexed ones. 
    # For demo, we use a temporary file if needed, but here we assume 'image' is a PIL object
    # from the gallery.
    temp_path = "data/processed/temp_explain.jpg"
    image.save(temp_path)
    
    saliency = memory_core.explain_retrieval(query, temp_path)
    # Overlay heatmap
    image_np = np.array(image)
    overlayed = overlay_heatmap(image_np, torch.tensor(saliency))
    return Image.fromarray(overlayed)

# 2. Build Gradio Interface
with gr.Blocks(title="Multimodal Memory AI") as demo:
    gr.Markdown("# 🧠 Multimodal Memory AI: Research Dashboard")
    gr.Markdown("Search through personal memories (CUB Birds) using semantic queries, visualize attention, and use RAG.")
    
    with gr.Tab("Semantic Search"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(label="Enter Search Query (e.g., 'a bird with a red head')", placeholder="Type here...")
                mode_radio = gr.Radio(["Text-to-Image", "Image-to-Image"], value="Text-to-Image", label="Search Mode")
                search_btn = gr.Button("Search Memories", variant="primary")
            with gr.Column(scale=1):
                k_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top K")
        
        gallery = gr.Gallery(label="Retrieved Memories", show_label=True, columns=3, height="400px")
        caption_output = gr.Code(label="Metadata Details", language="markdown")

    with gr.Tab("Conversational Memory (RAG)"):
        rag_input = gr.Textbox(label="Ask about your memories")
        rag_btn = gr.Button("Ask AI")
        rag_output = gr.Textbox(label="AI Response", lines=5)
        with gr.Accordion("See Prompt Context", open=False):
            prompt_viewer = gr.Code(language="text")

    with gr.Tab("Explainability (Saliency)"):
        gr.Markdown("Visualizing the semantic alignment between query and image.")
        with gr.Row():
            exp_query = gr.Textbox(label="Query to Explain")
            exp_image = gr.Image(label="Base Image", type="pil")
        explain_btn = gr.Button("Generate Saliency Map")
        exp_output = gr.Image(label="Saliency / Attention Overlay")

    # Wire up logic
    search_btn.click(
        multimodal_search, 
        inputs=[query_input, mode_radio, k_slider], 
        outputs=[gallery, caption_output]
    )
    
    rag_btn.click(
        generate_rag_answer,
        inputs=[rag_input],
        outputs=[rag_output, prompt_viewer]
    )
    
    explain_btn.click(
        explain_result,
        inputs=[exp_query, exp_image],
        outputs=[exp_output]
    )

if __name__ == "__main__":
    # Note: Install gradio first: pip install gradio
    demo.launch(share=False)
