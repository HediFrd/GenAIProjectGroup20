# scripts/download_ffhq_makeup.py
from huggingface_hub import snapshot_download
import os

def download_ffhq_makeup():
    """Download FFHQ-Makeup dataset from HuggingFace"""
    
    print("Downloading FFHQ-Makeup dataset...")
    print("This dataset contains 18,000 identities × 6 images (1 bare + 5 makeup styles)")
    print("Total: 108,000 images at 512×512 resolution")
    print("Size: ~15-20 GB")
    
    # Download from HuggingFace
    dataset_path = snapshot_download(
        repo_id="cyberagent/FFHQ-Makeup",
        repo_type="dataset",
        local_dir="data/ffhq_makeup",
        cache_dir="./cache"
    )
    
    print(f"\nDataset downloaded to: {dataset_path}")
    print("\nDataset structure:")
    print("- Each identity folder contains:")
    print("  - bare.jpg (original face)")
    print("  - makeup_01.jpg to makeup_05.jpg (5 makeup styles)")
    
    return dataset_path

if __name__ == "__main__":
    download_ffhq_makeup()
