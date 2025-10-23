# scripts/create_synthetic_pairs.py
import os
import sys
import json
import torch
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
import glob

class SyntheticDataGenerator:
    """Generate synthetic makeup training pairs using simple transformations"""
    
    def __init__(self, base_images_dirs, instruction_bank_path, output_dir):
        self.base_images_dirs = base_images_dirs if isinstance(base_images_dirs, list) else [base_images_dirs]
        self.output_dir = output_dir
        
        # Load instruction bank
        with open(instruction_bank_path, 'r') as f:
            self.instruction_bank = json.load(f)
        
        # Flatten instructions
        self.all_instructions = []
        for category, instructions in self.instruction_bank.items():
            for inst in instructions:
                self.all_instructions.append({
                    'category': category,
                    'text': inst
                })
        
        print(f"Loaded {len(self.all_instructions)} instructions")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/base", exist_ok=True)
        os.makedirs(f"{output_dir}/edited", exist_ok=True)
    
    def find_images(self):
        """Find all images in base directories"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_images = []
        
        print(f"Searching for images in: {self.base_images_dirs}")
        
        for base_dir in self.base_images_dirs:
            if not os.path.exists(base_dir):
                print(f"Warning: {base_dir} does not exist, skipping...")
                continue
            
            print(f"  Checking {base_dir}...")
            for ext in image_extensions:
                # Recursive search
                images = glob.glob(f"{base_dir}/**/{ext}", recursive=True)
                all_images.extend(images)
                
                # Non-recursive search
                images = glob.glob(f"{base_dir}/{ext}")
                all_images.extend(images)
        
        # Remove duplicates
        all_images = list(set(all_images))
        print(f"Found {len(all_images)} total images")
        return all_images
    
    def simple_makeup_transform(self, image, instruction_category):
        """
        Simple makeup transformation using color adjustments
        This is a placeholder - you can enhance this or integrate Stable-Makeup
        """
        img_array = np.array(image).astype(float)
        h, w = img_array.shape[:2]
        
        # Simple transformations based on category
        if instruction_category == "lips":
            # Enhance red channel in bottom portion (lips area)
            mask = np.zeros((h, w))
            center_h = int(h * 0.65)
            center_w = int(w * 0.5)
            radius = int(min(h, w) * 0.12)
            
            y, x = np.ogrid[:h, :w]
            lip_mask = ((y - center_h)**2 + (x - center_w)**2) <= radius**2
            mask[lip_mask] = 1.0
            
            # Add red tint
            img_array[:,:,0] = np.clip(img_array[:,:,0] + mask * 40, 0, 255)
            img_array[:,:,1] = np.clip(img_array[:,:,1] - mask * 10, 0, 255)
            img_array[:,:,2] = np.clip(img_array[:,:,2] - mask * 10, 0, 255)
            
        elif instruction_category == "eyes":
            # Darken eye region
            eye_mask = np.zeros((h, w))
            eye_y = int(h * 0.4)
            eye_mask[eye_y-int(h*0.1):eye_y+int(h*0.05), :] = 0.4
            img_array = np.clip(img_array * (1 - eye_mask[:,:,np.newaxis]), 0, 255)
            
            # Add some shimmer (brighten certain pixels)
            shimmer = np.random.random((h, w)) > 0.95
            shimmer_mask = shimmer * eye_mask
            img_array[:,:,0] = np.clip(img_array[:,:,0] + shimmer_mask * 50, 0, 255)
            img_array[:,:,1] = np.clip(img_array[:,:,1] + shimmer_mask * 50, 0, 255)
            img_array[:,:,2] = np.clip(img_array[:,:,2] + shimmer_mask * 50, 0, 255)
            
        elif instruction_category == "skin":
            # Slight warmth adjustment for foundation
            img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.08, 0, 255)
            img_array[:,:,1] = np.clip(img_array[:,:,1] * 1.05, 0, 255)
            img_array[:,:,2] = np.clip(img_array[:,:,2] * 0.98, 0, 255)
            
        elif instruction_category == "brows":
            # Darken eyebrow region
            brow_mask = np.zeros((h, w))
            brow_y = int(h * 0.35)
            brow_mask[brow_y-int(h*0.03):brow_y+int(h*0.03), int(w*0.2):int(w*0.45)] = 0.3
            brow_mask[brow_y-int(h*0.03):brow_y+int(h*0.03), int(w*0.55):int(w*0.8)] = 0.3
            img_array = np.clip(img_array * (1 - brow_mask[:,:,np.newaxis]), 0, 255)
            
        elif instruction_category == "overall":
            # Combination of subtle adjustments
            img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.05, 0, 255)
            img_array[:,:,1] = np.clip(img_array[:,:,1] * 1.03, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def generate_pairs(self, num_base_images=100, instructions_per_image=5):
        """Generate synthetic base-instruction-target triplets"""
        
        # Find all images
        image_files = self.find_images()
        
        if len(image_files) == 0:
            print("\nERROR: No images found!")
            print("Please ensure images exist in one of these directories:")
            for d in self.base_images_dirs:
                print(f"  - {d}")
            print("\nYou may need to:")
            print("1. Download FFHQ dataset first")
            print("2. Run: python scripts/download_ffhq_huggingface.py")
            return []
        
        # Limit to requested number
        if len(image_files) > num_base_images:
            image_files = random.sample(image_files, num_base_images)
        else:
            print(f"Note: Only {len(image_files)} images available (requested {num_base_images})")
        
        dataset = []
        pair_id = 0
        
        for img_file in tqdm(image_files, desc="Generating synthetic pairs"):
            try:
                base_image = Image.open(img_file).convert('RGB')
                
                # Resize to 512x512 if needed
                if base_image.size != (512, 512):
                    base_image = base_image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Generate multiple variations with different instructions
                for _ in range(instructions_per_image):
                    # Sample random instruction
                    instruction = random.choice(self.all_instructions)
                    
                    # Generate edited image
                    edited_image = self.simple_makeup_transform(
                        base_image.copy(),
                        instruction['category']
                    )
                    
                    # Save images
                    base_save_path = f"{self.output_dir}/base/pair_{pair_id:06d}.png"
                    edited_save_path = f"{self.output_dir}/edited/pair_{pair_id:06d}.png"
                    
                    base_image.save(base_save_path)
                    edited_image.save(edited_save_path)
                    
                    # Record metadata
                    dataset.append({
                        'pair_id': pair_id,
                        'base_image': base_save_path,
                        'edited_image': edited_save_path,
                        'instruction': instruction['text'],
                        'category': instruction['category'],
                        'source_image': img_file
                    })
                    
                    pair_id += 1
                    
            except Exception as e:
                print(f"\nError processing {img_file}: {e}")
                continue
        
        # Save dataset metadata
        metadata_path = f"{self.output_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\nGenerated {len(dataset)} training pairs")
        print(f"Saved to {self.output_dir}")
        print(f"Metadata saved to {metadata_path}")
        
        # Create train/val split
        val_split = int(len(dataset) * 0.1)
        train_data = dataset[val_split:]
        val_data = dataset[:val_split]
        
        # Save splits
        with open(f"{self.output_dir}/train_metadata.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(f"{self.output_dir}/val_metadata.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        return dataset

def main():
    # Search in multiple possible directories
    base_images_dirs = [
        "data/ffhq_raw",
        "data/ffhq_makeup",
        "data/ffhq_makeup/images",
        "data/ffhq_makeup/train",
    ]
    
    generator = SyntheticDataGenerator(
        base_images_dirs=base_images_dirs,
        instruction_bank_path="data/instruction_bank.json",
        output_dir="data/stage2_synthetic"
    )
    
    # Generate dataset
    # Start with small number for testing
    dataset = generator.generate_pairs(
        num_base_images=100,  # Use 100 images
        instructions_per_image=5  # Generate 5 variations per image
    )
    
    if len(dataset) > 0:
        print(f"\n✓ Success! Total pairs generated: {len(dataset)}")
        print(f"✓ You can now proceed to training")
    else:
        print(f"\n✗ No pairs generated - please download images first")
        print("\nQuick fix: Run one of these to download images:")
        print("  python scripts/download_ffhq_huggingface.py")
        print("  python scripts/download_ffhq_makeup.py")

if __name__ == "__main__":
    main()
