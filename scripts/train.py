import torch
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

# Internal modules
from data.loaders.multimodal_dataset import MultimodalDataset
from models.encoders.multimodal_encoders import VisionEncoder, TextEncoder, ContextualEncoder
from models.fusion.cross_modal_fusion import MultimodalFusionTransformer
from models.alignment.contrastive_loss import InfoNCELoss

def train_one_epoch(vision_encoder, text_encoder, dataloader, optimizer, loss_fn, device, fusion_model=None):
    vision_encoder.train()
    text_encoder.train()
    total_loss = 0
    
    print(f"Momentum Queue status: ptr={loss_fn.queue_ptr.item()}/{loss_fn.queue_size}", flush=True)
    
    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        meta_vecs = batch.get('metadata_vec').to(device) if 'metadata_vec' in batch else None
        
        optimizer.zero_grad()
        
        # Forward pass
        if fusion_model:
            # Extract patches and tokens
            img_seq = vision_encoder(images, return_sequence=True)
            txt_seq = text_encoder(input_ids, attention_mask, return_sequence=True)
            
            # Cross-modal attention
            fused_img, fused_txt = fusion_model(img_seq, txt_seq)
            
            # Pool fused representations
            image_embeddings = vision_encoder.projection(fused_img[:, 0])
            
            mask = attention_mask.unsqueeze(-1).expand(fused_txt.size()).float()
            text_embeddings = text_encoder.projection(torch.sum(fused_txt * mask, 1) / torch.clamp(mask.sum(1), min=1e-9))
        else:
            image_embeddings = vision_encoder(images, metadata_vec=meta_vecs)
            text_embeddings = text_encoder(input_ids, attention_mask, metadata_vec=meta_vecs)
        
        # Calculate loss (now uses Global Hard Negative Mining via MoCo Queue)
        loss = loss_fn(image_embeddings, text_embeddings)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Momentum Queue update (Must happen after backward pass)
        if hasattr(loss_fn, 'update_queue'):
            loss_fn.update_queue(text_embeddings)
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Batch [{i}/{len(dataloader)}], Loss: {loss.item():.4f}", flush=True)
            
    return total_loss / len(dataloader)

def main(config_overrides=None):
    # Research Parameters (Ablation Configuration)
    CONFIG = {
        "embedding_dim": 512,
        "batch_size": 16, # Reduced batch size for CPU stability
        "lr": 5e-5,
        "epochs": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "text_model": "bert-base-uncased",
        "vision_model": "vit_b_16",
        "temperature": 0.07,
        "queue_size": 4096,
        "max_samples": 1000, # Subsetting for demonstration
        "use_fusion": False, # Advanced cross-modal attention
        "use_context": False, # Use metadata-driven alignment
        "experiment_name": "contrastive_moco_finetune"
    }
    
    if config_overrides:
        CONFIG.update(config_overrides)
    
    device = torch.device(CONFIG["device"])
    print(f"--- Fine-Tuning Experiment: {CONFIG['experiment_name']} ---", flush=True)
    print(f"Backbone: {CONFIG['vision_model']} + {CONFIG['text_model']}", flush=True)
    print(f"Mining Strategy: Global Momentum Queue (Size: {CONFIG['queue_size']})", flush=True)

    # Initializing Encoders
    vision_encoder = VisionEncoder(model_name=CONFIG["vision_model"], embedding_dim=CONFIG["embedding_dim"]).to(device)
    text_encoder = TextEncoder(model_name=CONFIG["text_model"], embedding_dim=CONFIG["embedding_dim"]).to(device)
    
    # Advanced Contextual Wrapping
    if CONFIG["use_context"]:
        print("Enabling Contextual Multi-modal Alignment (Metadata-driven)...", flush=True)
        vision_encoder = ContextualEncoder(vision_encoder, embedding_dim=CONFIG["embedding_dim"]).to(device)
        text_encoder = ContextualEncoder(text_encoder, embedding_dim=CONFIG["embedding_dim"]).to(device)
    
    # Optimizer Params
    params = list(vision_encoder.parameters()) + list(text_encoder.parameters())

    fusion_model = None
    if CONFIG["use_fusion"]:
        print("Initializing Cross-Modal Fusion Transformer...", flush=True)
        fusion_model = MultimodalFusionTransformer(embed_dim=CONFIG["embedding_dim"]).to(device)
        params += list(fusion_model.parameters())
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])
    
    # Optimizer and Loss
    optimizer = optim.AdamW(params, lr=CONFIG["lr"], weight_decay=0.01)
    loss_fn = InfoNCELoss(temperature=CONFIG["temperature"], queue_size=CONFIG["queue_size"]).to(device)
    
    # Data Loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset paths (should be passed via config or environment)
    DATA_PATH = os.environ.get("DATASET_PATH", "data/raw")
    IMG_DIR = os.path.join(DATA_PATH)
    CSV_FILE = os.path.join(DATA_PATH, "metadata.csv")
    
    # Override from config if available
    if config_overrides and "dataset" in config_overrides:
        ds_config = config_overrides["dataset"]
        DATA_PATH = ds_config.get("path", DATA_PATH)
        IMG_DIR = os.path.join(DATA_PATH, ds_config.get("images_dir", "Images"))
        CSV_FILE = os.path.join(DATA_PATH, ds_config.get("captions_file", "captions.txt"))
    
    try:
        dataset = MultimodalDataset(CSV_FILE, IMG_DIR, transform=transform, tokenizer=tokenizer)
        if CONFIG["max_samples"] and len(dataset) > CONFIG["max_samples"]:
            print(f"Subsetting dataset to {CONFIG['max_samples']} samples.", flush=True)
            indices = list(range(CONFIG["max_samples"]))
            dataset = torch.utils.data.Subset(dataset, indices)
            
        dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
        print(f"Loaded dataset: {len(dataset)} entries.", flush=True)
    except Exception as e:
        print(f"Data loading failed: {e}. Fine-tuning aborted.", flush=True)
        return None

    # Training Loop
    checkpoint_dir = "data/processed/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Starting fine-tuning on {device}...", flush=True)
    for epoch in range(CONFIG["epochs"]):
        avg_loss = train_one_epoch(vision_encoder, text_encoder, dataloader, optimizer, loss_fn, device, fusion_model=fusion_model)
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Average Loss: {avg_loss:.4f}", flush=True)
        
        # Save Research Checkpoints
        save_path_v = f"{checkpoint_dir}/vision_encoder_moco.pth"
        save_path_t = f"{checkpoint_dir}/text_encoder_moco.pth"
        
        torch.save(vision_encoder.state_dict(), save_path_v)
        torch.save(text_encoder.state_dict(), save_path_t)
        print(f"Fine-tuned weights saved to {checkpoint_dir}", flush=True)

    print(f"Fine-tuning session '{CONFIG['experiment_name']}' completed.", flush=True)
    
    return {
        "vision_encoder": vision_encoder,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "config": CONFIG
    }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Fine-tune Multimodal Memory AI")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    args = parser.parse_args()
    
    config_overrides = None
    if args.config:
        with open(args.config, 'r') as f:
            config_overrides = json.load(f)
            
    main(config_overrides=config_overrides)
