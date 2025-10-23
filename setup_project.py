# setup_project.py
import os

def create_project_structure():
    """Create the complete project directory structure"""
    directories = [
        'data/ffhq_raw',
        'data/ffhq_makeup',
        'data/stage1_instructions',
        'data/stage2_synthetic',
        'models/stable_diffusion_v15',
        'models/stage1_model',
        'models/stage2_model',
        'models/blip2',
        'models/stable_makeup',
        'scripts',
        'outputs/stage1_results',
        'outputs/stage2_results',
        'configs',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
