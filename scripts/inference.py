# scripts/inference.py
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from PIL import Image
import os
import glob
import urllib.request
from io import BytesIO

def modify_unet_for_instruct_pix2pix(unet):
    """Modify UNet to accept 8 input channels instead of 4"""
    conv_in = unet.conv_in
    
    new_conv_in = torch.nn.Conv2d(
        8,
        conv_in.out_channels,
        kernel_size=conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding,
        bias=conv_in.bias is not None
    )
    
    with torch.no_grad():
        new_conv_in.weight[:, :4, :, :] = conv_in.weight.clone()
        new_conv_in.weight[:, 4:, :, :] = torch.zeros_like(conv_in.weight)
        
        if conv_in.bias is not None:
            new_conv_in.bias.copy_(conv_in.bias)
    
    new_conv_in = new_conv_in.to(conv_in.weight.device, dtype=conv_in.weight.dtype)
    unet.conv_in = new_conv_in
    
    # UPDATE THE CONFIG - This is the key fix!
    unet.config.in_channels = 8
    
    return unet


def load_model(model_path="models/stage2_model", use_fp16=True):
    """Load your trained model with 8-channel UNet"""
    print(f"Loading model from {model_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if use_fp16 and torch.cuda.is_available() else torch.float32
    
    # Load VAE, text encoder, tokenizer from original SD
    print("Loading VAE, text encoder, and tokenizer...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae",
        torch_dtype=dtype
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder",
        torch_dtype=dtype
    )
    
    # Load base UNet and modify for 8 channels
    print("Loading UNet and modifying for 8 channels...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=dtype
    )
    
    # Modify for 8 channels
    unet = modify_unet_for_instruct_pix2pix(unet)
    
    # Load trained weights
    print(f"Loading trained weights from {model_path}...")
    unet_path = os.path.join(model_path, "unet")
    
    # Load state dict
    safetensors_files = glob.glob(os.path.join(unet_path, "*.safetensors"))
    
    if safetensors_files:
        from safetensors.torch import load_file
        state_dict = {}
        for file in safetensors_files:
            state_dict.update(load_file(file))
    else:
        bin_files = glob.glob(os.path.join(unet_path, "*.bin"))
        state_dict = {}
        for file in bin_files:
            state_dict.update(torch.load(file, map_location='cpu'))
    
    # Load weights into UNet
    unet.load_state_dict(state_dict, strict=True)
    print("✅ Trained weights loaded!")
    
    # Load scheduler
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    
    # Create pipeline
    pipe = StableDiffusionInstructPix2PixPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    
    pipe.to(device)
    print(f"✅ Model loaded successfully on {device}!")
    
    return pipe

def load_image_from_path_or_url(source):
    """Load image from file path or URL"""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"Downloading image from URL...")
        try:
            with urllib.request.urlopen(source) as response:
                img_data = response.read()
            image = Image.open(BytesIO(img_data)).convert("RGB")
            print("✅ Image downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download image: {e}")
            raise
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image not found: {source}")
        image = Image.open(source).convert("RGB")
    
    return image

def get_test_image():
    """Get a test image - try local first, then download"""
    # Check local paths first
    local_paths = [
        "test_images/face.png",
        "test_images/face.jpg",
        "data/stage1_instructions/00000_original.png",
        "data/stage2_synthetic/00000_original.png",
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            print(f"Using local test image: {path}")
            return path
    
    # Try to download
    print("No local test image found. Attempting to download...")
    os.makedirs("test_images", exist_ok=True)
    
    # Try different sources
    test_urls = [
        "https://i.pravatar.cc/512?img=12",
        "https://randomuser.me/api/portraits/women/44.jpg",
        "https://randomuser.me/api/portraits/men/32.jpg",
    ]
    
    for i, url in enumerate(test_urls):
        try:
            print(f"Trying source {i+1}...")
            img = load_image_from_path_or_url(url)
            save_path = "test_images/face.png"
            img.save(save_path)
            print(f"✅ Test image saved to: {save_path}")
            return save_path
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    # If all fail, provide instructions
    print("\n❌ Could not download test image.")
    print("\nPlease manually add a test image:")
    print("1. Copy any face photo to: test_images/face.png")
    print("2. Or run: python scripts/inference.py <path_to_your_image.jpg>")
    print("3. Or run: python scripts/inference.py <image_url>")
    exit(1)

def apply_makeup(pipe, image_source, instruction, output_path=None, **kwargs):
    """
    Apply makeup to an image based on text instruction
    
    Args:
        pipe: Loaded pipeline
        image_source: Path to input image or URL
        instruction: Text instruction (e.g., "Apply red lipstick")
        output_path: Where to save result (optional)
        **kwargs: Additional parameters for the pipeline
    """
    # Load image from path or URL
    image = load_image_from_path_or_url(image_source)
    original_size = image.size
    
    # Resize to match training resolution
    image = image.resize((256, 256))
    
    # Default parameters (you can override with kwargs)
    params = {
        'num_inference_steps': 50,
        'image_guidance_scale': 1.5,  # How much to preserve original (1.0-2.0)
        'guidance_scale': 7.5,  # How much to follow text (5.0-10.0)
    }
    params.update(kwargs)
    
    print(f"\nApplying: {instruction}")
    print(f"Parameters: steps={params['num_inference_steps']}, img_guidance={params['image_guidance_scale']}, text_guidance={params['guidance_scale']}")
    
    # Generate result
    with torch.no_grad():
        result = pipe(
            prompt=instruction,
            image=image,
            **params
        ).images[0]
    
    # Resize back to original size
    result = result.resize(original_size, Image.LANCZOS)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result.save(output_path)
        print(f"✅ Saved to: {output_path}")
    
    return result

def batch_inference(pipe, image_source, instructions, output_dir="outputs/inference"):
    """Run multiple instructions on the same image"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, instruction in enumerate(instructions):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(instructions)}]")
        output_path = f"{output_dir}/{i:02d}_{instruction.replace(' ', '_')[:30]}.png"
        result = apply_makeup(pipe, image_source, instruction, output_path)
        results.append(result)
    
    return results

def create_comparison_grid(original_path, results, instructions, output_path):
    """Create a grid showing original + all results"""
    original = load_image_from_path_or_url(original_path).resize((256, 256))
    
    # Calculate grid size
    n_results = len(results)
    cols = min(4, n_results + 1)  # Max 4 columns
    rows = (n_results + cols) // cols
    
    # Create grid
    grid_width = cols * 256
    grid_height = rows * 256
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Paste images
    grid.paste(original, (0, 0))
    
    for i, (result, instruction) in enumerate(zip(results, instructions)):
        x = ((i + 1) % cols) * 256
        y = ((i + 1) // cols) * 256
        result_resized = result.resize((256, 256))
        grid.paste(result_resized, (x, y))
    
    grid.save(output_path)
    print(f"\n✅ Comparison grid saved to: {output_path}")
    
    return grid

# Example usage
if __name__ == "__main__":
    import sys
    
    # Configuration
    MODEL_PATH = "models/stage2_model"  # Change to "models/stage1_model" to test Stage 1
    OUTPUT_DIR = "outputs/inference_results"
    
    # Get test image
    if len(sys.argv) > 1:
        # Image provided as argument
        TEST_IMAGE = sys.argv[1]
        print(f"Using provided image: {TEST_IMAGE}")
    else:
        # Auto-detect or download
        TEST_IMAGE = get_test_image()
    
    print("\n" + "="*60)
    print("AI MAKEUP INFERENCE")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Test Image: {TEST_IMAGE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Load model
    pipe = load_model(MODEL_PATH, use_fp16=True)
    
    # Test single instruction
    print("\n" + "="*60)
    print("SINGLE INFERENCE TEST")
    print("="*60)
    result = apply_makeup(
        pipe,
        TEST_IMAGE,
        "Apply red lipstick",
        f"{OUTPUT_DIR}/single_test.png"
    )
    
    # Test multiple instructions
    print("\n" + "="*60)
    print("BATCH INFERENCE TEST")
    print("="*60)
    
    instructions = [
        "Apply red lipstick",
        "Add winged eyeliner",
        "Apply natural makeup",
        "Add dramatic smokey eyes",
        "Apply pink blush",
        "Add false eyelashes",
        "Apply nude lipstick",
        "Make eyes brighter",
        "Apply burgundy lipstick",
        "Add cat eye makeup"
    ]
    
    results = batch_inference(pipe, TEST_IMAGE, instructions, OUTPUT_DIR)
    
    # Create comparison grid
    print("\n" + "="*60)
    print("CREATING COMPARISON GRID")
    print("="*60)
    create_comparison_grid(
        TEST_IMAGE, 
        results, 
        instructions, 
        f"{OUTPUT_DIR}/comparison_grid.png"
    )
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"Generated {len(results)} images")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"Comparison grid: {OUTPUT_DIR}/comparison_grid.png")
    print("="*60)
