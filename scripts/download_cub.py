import kagglehub
import os

def download_cub():
    print("Initializing download of CUB-200-2011 (Fine-grained Research Dataset)...")
    try:
        # Download latest version from Kaggle
        path = kagglehub.dataset_download("veeralakrishna/200-bird-species-with-11788-images")
        print("\n" + "="*50)
        print("SUCCESS: CUB-200-2011 Downloaded.")
        print(f"Path to dataset files: {path}")
        print("="*50)
        
        # Display directory structure to help with loader setup
        print("\nDataset Contents:")
        for item in os.listdir(path):
            print(f"- {item}")
            
    except Exception as e:
        print(f"Error during download: {e}")

if __name__ == "__main__":
    download_cub()
