# scripts/setup_stable_makeup.py
import os
import subprocess
import torch

def setup_stable_makeup():
    """Clone and setup Stable-Makeup model"""
    
    stable_makeup_dir = "models/stable_makeup"
    
    if not os.path.exists(os.path.join(stable_makeup_dir, "infer_kps.py")):
        print("Cloning Stable-Makeup repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/Xiaojiu-z/Stable-Makeup.git",
            stable_makeup_dir
        ])
    
    print("Stable-Makeup setup complete!")
    print("\nNext steps:")
    print("1. Download pretrained weights from Google Drive")
    print("2. Place them in models/stable_makeup/models/stablemakeup/")
    
if __name__ == "__main__":
    setup_stable_makeup()
