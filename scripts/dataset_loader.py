# scripts/dataset_loader.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torchvision import transforms

class InstructPix2PixDataset(Dataset):
    """Dataset for InstructPix2Pix training"""
    
    def __init__(self, data_dir, resolution=512, is_stage1=True):
        self.data_dir = data_dir
        self.resolution = resolution
        self.is_stage1 = is_stage1
        
        # Load metadata
        if is_stage1:
            metadata_path = os.path.join(data_dir, "instructions.json")
        else:
            metadata_path = os.path.join(data_dir, "metadata.json")
        
        with open(metadata_path, 'r') as f:
            self.data = json.load(f)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.is_stage1:
            # Stage 1: Same image as input/output with instruction
            image_path = item['image_path']
            instruction = item['instructions']['instruction_variations'][0]
            
            image = Image.open(image_path).convert('RGB')
            original_image = self.transform(image)
            edited_image = original_image.clone()
            
        else:
            # Stage 2: Different base and edited images
            base_image = Image.open(item['base_image']).convert('RGB')
            edited_image = Image.open(item['edited_image']).convert('RGB')
            instruction = item['instruction']
            
            original_image = self.transform(base_image)
            edited_image = self.transform(edited_image)
        
        return {
            'original_image': original_image,
            'edited_image': edited_image,
            'instruction': instruction
        }
