import torch
import os
import json
import pandas as pd
from scripts.train import main as train_fn
from scripts.evaluate import evaluate_retrieval
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from data.loaders.multimodal_dataset import MultimodalDataset

def run_ablation_study():
    """
    Automated research tool to compare different project configurations.
    Generates a comparative analysis of baseline vs. advanced models.
    """
    print("\n--- [Multimodal Memory AI: AUTOMATED ABLATION STUDY] ---")
    
    # Define experiment configurations
    experiments = [
        {
            "name": "baseline",
            "config": {
                "use_fusion": False,
                "queue_size": 0, # Disable HNM
                "epochs": 1,
                "max_samples": 500
            }
        },
        {
            "name": "hnm_only",
            "config": {
                "use_fusion": False,
                "queue_size": 1024,
                "epochs": 1,
                "max_samples": 500
            }
        },
        {
            "name": "deep_fusion_v2",
            "config": {
                "use_fusion": True,
                "queue_size": 1024,
                "epochs": 1,
                "max_samples": 500
            }
        }
    ]
    
    results_summary = []
    
    # Common setup for evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "data/raw"
    CSV_FILE = os.path.join(DATA_PATH, "metadata.csv")
    IMG_DIR = DATA_PATH
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load test set (subset of CUB)
    test_dataset = MultimodalDataset(CSV_FILE, IMG_DIR, transform=transform, tokenizer=tokenizer)
    # Use different indices than training if needed, here we just take a slice
    test_indices = list(range(110000, 110500)) 
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)

    for exp in experiments:
        print(f"\n>> Running Experiment: {exp['name']}...")
        
        # 1. Train the model with specific config
        trained_bundle = train_fn(config_overrides=exp['config'])
        
        # 2. Evaluate
        print(f">> Evaluating {exp['name']}...")
        metrics = evaluate_retrieval(
            trained_bundle['vision_encoder'],
            trained_bundle['text_encoder'],
            test_loader,
            device
        )
        
        # 3. Store results
        res = {
            "Experiment": exp['name'],
            "I2T_R@1": metrics['I2T_Recall']['Recall@1'],
            "I2T_R@5": metrics['I2T_Recall']['Recall@5'],
            "T2I_R@1": metrics['T2I_Recall']['Recall@1'],
            "T2I_R@5": metrics['T2I_Recall']['Recall@5']
        }
        results_summary.append(res)
        print(f"Results: R@1={res['I2T_R@1']:.2f}")

    # 4. Generate Research Report
    df = pd.DataFrame(results_summary)
    report_path = "reports/ablation_study_results.csv"
    df.to_csv(report_path, index=False)
    
    print("\n" + "="*50)
    print("ABLATION STUDY COMPLETED")
    print(f"Comparative report saved to: {report_path}")
    print("="*50)
    print(df.to_string())

if __name__ == "__main__":
    run_ablation_study()
