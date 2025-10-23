# scripts/generate_instructions.py
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
import glob

class InstructionGenerator:
    """Generate makeup instructions using BLIP-2 (no API key needed)"""
    
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        """Initialize BLIP-2 model from HuggingFace"""
        print("Loading BLIP-2 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Makeup-specific prompts for BLIP-2
        self.makeup_prompts = [
            "Question: Describe the makeup style in detail. Answer:",
            "Question: What makeup is applied on the eyes? Answer:",
            "Question: What makeup is applied on the lips? Answer:",
            "Question: Describe the skin makeup and foundation. Answer:",
            "Question: What is the overall makeup look? Answer:"
        ]
        
    def generate_caption(self, image_path, prompt):
        """Generate caption for single image with specific prompt"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(
            self.device, torch.float16 if self.device == "cuda" else torch.float32
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        
        return generated_text
    
    def create_instruction_variations(self, base_description):
        """Create instruction variations without using external APIs"""
        # Template-based instruction generation
        templates = [
            "Apply {} makeup style",
            "Transform the face with {} makeup",
            "Add {} makeup to the face",
            "Create a {} makeup look",
            "Give a {} makeup appearance"
        ]
        
        variations = []
        for template in templates:
            variations.append(template.format(base_description))
        
        return variations
    
    def generate_detailed_instructions(self, image_path):
        """Generate multiple detailed instructions for a single image"""
        instructions = {}
        
        # Generate captions for different aspects
        for i, prompt in enumerate(self.makeup_prompts):
            caption = self.generate_caption(image_path, prompt)
            instructions[f'aspect_{i}'] = caption
        
        # Create combined description
        combined = f"{instructions['aspect_0']}. Eyes: {instructions['aspect_1']}. " \
                   f"Lips: {instructions['aspect_2']}. Skin: {instructions['aspect_3']}."
        
        # Generate variations
        variations = self.create_instruction_variations(instructions['aspect_4'])
        
        return {
            'detailed_aspects': instructions,
            'combined_description': combined,
            'instruction_variations': variations
        }
    
    def find_images(self, base_dirs):
        """Find all images in given directories"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_images = []
        
        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                print(f"Warning: {base_dir} does not exist, skipping...")
                continue
                
            for ext in image_extensions:
                # Recursive search
                images = glob.glob(f"{base_dir}/**/{ext}", recursive=True)
                all_images.extend(images)
                
                # Non-recursive search
                images = glob.glob(f"{base_dir}/{ext}")
                all_images.extend(images)
        
        # Remove duplicates
        all_images = list(set(all_images))
        return all_images
    
    def process_dataset(self, input_dirs, output_file, limit=None):
        """Process entire dataset and generate instructions"""
        
        # Find all images
        print(f"Searching for images in: {input_dirs}")
        image_files = self.find_images(input_dirs)
        
        print(f"Found {len(image_files)} total images")
        
        if len(image_files) == 0:
            print("\nERROR: No images found!")
            print("Please check that images exist in one of these directories:")
            for d in input_dirs:
                print(f"  - {d}")
            return []
        
        if limit:
            print(f"Processing first {limit} images...")
            image_files = image_files[:limit]
        
        results = []
        
        for img_path in tqdm(image_files, desc="Generating instructions"):
            try:
                instructions = self.generate_detailed_instructions(img_path)
                
                result = {
                    'image_path': img_path,
                    'filename': os.path.basename(img_path),
                    'instructions': instructions
                }
                results.append(result)
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved {len(results)} instruction sets to {output_file}")
        return results

def main():
    # Initialize generator
    generator = InstructionGenerator()
    
    # Look in multiple possible directories
    input_dirs = [
        "data/ffhq_raw",
        "data/ffhq_makeup",
        "data/ffhq_makeup/images",
        "data/ffhq_makeup/train",
    ]
    
    output_file = "data/stage1_instructions/instructions.json"
    
    # Generate instructions (set limit for testing, remove for full dataset)
    results = generator.process_dataset(input_dirs, output_file, limit=10)
    
    if len(results) > 0:
        print(f"\n✓ Successfully generated instructions for {len(results)} images")
        print(f"✓ Output saved to: {output_file}")
        print("\nSample instruction:")
        print(json.dumps(results[0], indent=2))
    else:
        print("\n✗ No instructions generated - please check image directories")

if __name__ == "__main__":
    main()
