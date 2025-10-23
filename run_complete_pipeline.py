# run_complete_pipeline.py
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error in: {description}")
        sys.exit(1)

def main():
    print("Starting complete makeup editing project pipeline...")
    
    # Step 1: Setup project
    run_command(
        "python setup_project.py",
        "Step 1: Setting up project structure"
    )
    
    # Step 2: Create instruction bank
    run_command(
        "python scripts/create_instruction_bank.py",
        "Step 2: Creating fine-grained instruction bank"
    )
    
    # Step 3: Generate instructions for FFHQ
    run_command(
        "python scripts/generate_instructions.py",
        "Step 3: Generating instructions using BLIP-2"
    )
    
    # Step 4: Generate synthetic training pairs
    run_command(
        "python scripts/create_synthetic_pairs.py",
        "Step 4: Creating synthetic training pairs"
    )
    
    # Step 5: Train stage 1
    run_command(
        "python scripts/train_stage1.py",
        "Step 5: Training Stage 1 (Base model on FFHQ)"
    )
    
    # Step 6: Train stage 2
    run_command(
        "python scripts/train_stage2.py",
        "Step 6: Training Stage 2 (Fine-tuning on synthetic data)"
    )
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nFinal model saved to: models/stage2_model")
    print("\nTo run inference:")
    print("python scripts/inference.py --model models/stage2_model \\")
    print("    --input your_image.jpg \\")
    print("    --instruction 'Apply red lipstick' \\")
    print("    --output result.jpg")

if __name__ == "__main__":
    main()
