# scripts/train_stage1.py
import os
os.environ["XFORMERS_DISABLED"] = "1"

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
from dataset_loader import InstructPix2PixDataset
import glob

def modify_unet_for_instruct_pix2pix(unet):
    """Modify UNet to accept 8 input channels instead of 4"""
    conv_in = unet.conv_in
    
    new_conv_in = torch.nn.Conv2d(
        8,  # 8 input channels (4 noisy + 4 conditional)
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
    unet.config.in_channels = 8
    
    return unet

class Stage1Trainer:
    """Train InstructPix2Pix on FFHQ with generated instructions"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize accelerator with fp16 mixed precision for speed
        self.accelerator = Accelerator(
            mixed_precision=config['mixed_precision'],
            gradient_accumulation_steps=config['gradient_accumulation_steps']
        )
        
        # Check if resuming from checkpoint
        if config.get('resume_from_checkpoint'):
            print(f"Resuming from checkpoint: {config['resume_from_checkpoint']}")
            self.load_from_checkpoint(config['resume_from_checkpoint'])
        else:
            print("Loading Stable Diffusion v1.5...")
            model_id = config['base_model']
            
            # Load all components in float16 for maximum speed
            self.vae = AutoencoderKL.from_pretrained(
                model_id, 
                subfolder="vae",
                torch_dtype=torch.float16
            )
            
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer"
            )
            
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=torch.float16
            )
            
            self.unet = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                torch_dtype=torch.float16  # Use float16 for speed
            )
            
            # Modify UNet for InstructPix2Pix (8 input channels)
            print("Modifying UNet for InstructPix2Pix (4 -> 8 input channels)...")
            self.unet = modify_unet_for_instruct_pix2pix(self.unet)
            
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                model_id,
                subfolder="scheduler"
            )
        
        # Create pipeline from components
        self.pipe = StableDiffusionInstructPix2PixPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Only train UNet
        self.unet.train()
        
        # Optimizer with better settings for 30GB GPU
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Dataset
        self.train_dataset = InstructPix2PixDataset(
            config['data_dir'],
            resolution=config['resolution'],
            is_stage1=True
        )
        
        print(f"Training dataset size: {len(self.train_dataset)}")
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config['train_batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True  # Enable for faster data transfer
        )
        
        # Prepare with accelerator
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        
        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device)
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"GPU Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def load_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        print(f"Loading model components from {checkpoint_path}...")
        
        # Load VAE from original SD
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=torch.float16
        )
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder",
            torch_dtype=torch.float16
        )
        
        # Load base UNet and modify for 8 channels
        print("Loading base UNet and modifying for 8 channels...")
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=torch.float16
        )
        
        # Modify for 8 channels
        self.unet = modify_unet_for_instruct_pix2pix(self.unet)
        
        # Now load the trained weights
        print(f"Loading trained weights from {checkpoint_path}/unet...")
        checkpoint_unet_path = os.path.join(checkpoint_path, "unet")
        
        # Load the state dict
        safetensors_files = glob.glob(os.path.join(checkpoint_unet_path, "*.safetensors"))
        
        if safetensors_files:
            from safetensors.torch import load_file
            state_dict = {}
            for file in safetensors_files:
                state_dict.update(load_file(file))
        else:
            bin_files = glob.glob(os.path.join(checkpoint_unet_path, "*.bin"))
            state_dict = {}
            for file in bin_files:
                state_dict.update(torch.load(file, map_location='cpu'))
        
        # Load the state dict into the UNet
        self.unet.load_state_dict(state_dict, strict=True)
        print("Trained weights loaded successfully!")
        
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        
        print("Checkpoint loaded successfully!")
    
    def load_optimizer_state(self, checkpoint_path):
        """Load optimizer state if available"""
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            print(f"Loading optimizer state from {state_path}...")
            state = torch.load(state_path, map_location='cpu')
            try:
                self.optimizer.load_state_dict(state['optimizer'])
                self.lr_scheduler.load_state_dict(state['lr_scheduler'])
                print("Optimizer and scheduler state loaded successfully!")
            except Exception as e:
                print(f"Could not load optimizer state: {e}")
                print("Continuing with fresh optimizer state...")
        else:
            print("No optimizer state found, starting fresh...")
    
    def train(self, num_epochs, start_epoch=0):
        """Main training loop"""
        global_step = start_epoch * len(self.train_dataloader)
        best_loss = float('inf')
        
        print(f"\nStarting training from epoch {start_epoch} to {num_epochs}")
        print(f"Global step: {global_step}")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                disable=not self.accelerator.is_local_main_process
            )
            progress_bar.set_description(f"Epoch {epoch}/{num_epochs}")
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Images are already normalized by dataset
                    original_images = batch['original_image'].to(
                        self.accelerator.device, 
                        dtype=torch.float16
                    )
                    edited_images = batch['edited_image'].to(
                        self.accelerator.device, 
                        dtype=torch.float16
                    )
                    
                    # Encode images
                    with torch.no_grad():
                        latents_orig = self.vae.encode(original_images).latent_dist.sample()
                        latents_orig = latents_orig * self.vae.config.scaling_factor
                        
                        latents_edited = self.vae.encode(edited_images).latent_dist.sample()
                        latents_edited = latents_edited * self.vae.config.scaling_factor
                    
                    # Encode text
                    text_inputs = self.tokenizer(
                        batch['instruction'],
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        text_embeddings = self.text_encoder(
                            text_inputs.input_ids.to(self.accelerator.device)
                        )[0]
                    
                    # Sample noise
                    noise = torch.randn_like(latents_edited)
                    bsz = latents_edited.shape[0]
                    
                    # Sample timesteps
                    timesteps = torch.randint(
                        0, self.scheduler.config.num_train_timesteps,
                        (bsz,), device=latents_edited.device
                    )
                    timesteps = timesteps.long()
                    
                    # Add noise to edited latents
                    noisy_latents = self.scheduler.add_noise(
                        latents_edited, noise, timesteps
                    )
                    
                    # Concatenate with original latents (InstructPix2Pix conditioning)
                    latent_model_input = torch.cat([noisy_latents, latents_orig], dim=1)
                    
                    # Predict noise
                    model_pred = self.unet(
                        latent_model_input,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample
                    
                    # Calculate loss
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Check for NaN
                    if torch.isnan(loss):
                        print(f"\nWarning: NaN loss detected at step {global_step}, skipping...")
                        continue
                    
                    # Backpropagation
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update metrics
                current_loss = loss.detach().item()
                epoch_loss += current_loss
                
                progress_bar.update(1)
                logs = {
                    "loss": current_loss,
                    "avg_loss": epoch_loss / (step + 1),
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "step": global_step,
                    "epoch": epoch
                }
                progress_bar.set_postfix(**logs)
                global_step += 1
                
                # Save checkpoint
                if global_step % self.config['checkpointing_steps'] == 0:
                    self.save_checkpoint(global_step)
            
            progress_bar.close()
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"\nEpoch {epoch} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                print(f"New best loss: {best_loss:.4f} - Saving best model...")
                self.save_checkpoint("best")
            
            # Save at end of each epoch
            self.save_checkpoint(f"epoch_{epoch}")
            
        # Final save
        self.save_model()
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = f"{self.config['output_dir']}/checkpoint-{step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(
                os.path.join(checkpoint_dir, "unet")
            )
            
            # Also save optimizer and scheduler state
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, "training_state.pt"))
            
            print(f"\nSaved checkpoint to {checkpoint_dir}")
    
    def save_model(self):
        """Save final model"""
        if self.accelerator.is_main_process:
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            
            pipeline = StableDiffusionInstructPix2PixPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=unwrapped_unet,
                scheduler=self.scheduler,
                safety_checker=None,
                feature_extractor=None
            )
            
            pipeline.save_pretrained(self.config['output_dir'])
            print(f"\nSaved final model to {self.config['output_dir']}")

def main():
    config = {
        'base_model': 'runwayml/stable-diffusion-v1-5',
        'resume_from_checkpoint': None,  # Set to checkpoint path to resume
        'data_dir': 'data/stage1_instructions',
        'output_dir': 'models/stage1_model',
        'resolution': 512,  # Full resolution
        'train_batch_size': 8,  # Larger batch size for 30GB GPU
        'num_workers': 4,  # More workers for faster data loading
        'learning_rate': 1e-5,
        'gradient_accumulation_steps': 2,  # Less accumulation needed with larger batch
        'mixed_precision': 'fp16',  # Enable fp16 for speed
        'checkpointing_steps': 1000,
        'num_epochs': 100,
        'start_epoch': 0
    }
    
    print("="*70)
    if config.get('resume_from_checkpoint'):
        print("InstructPix2Pix Training - Stage 1 (RESUMING - 30GB GPU)")
    else:
        print("InstructPix2Pix Training - Stage 1 (30GB GPU Optimized)")
    print("="*70)
    if config.get('resume_from_checkpoint'):
        print(f"Resuming from: {config['resume_from_checkpoint']}")
        print(f"Starting epoch: {config['start_epoch']}")
    print(f"Resolution: {config['resolution']}x{config['resolution']}")
    print(f"Batch size: {config['train_batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Mixed precision: {config['mixed_precision']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Num workers: {config['num_workers']}")
    print("="*70)
    
    trainer = Stage1Trainer(config)
    
    # Load optimizer state if resuming
    if config.get('resume_from_checkpoint'):
        trainer.load_optimizer_state(config['resume_from_checkpoint'])
    
    trainer.train(num_epochs=config['num_epochs'], start_epoch=config['start_epoch'])
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Model saved to: {config['output_dir']}")
    print("="*70)

if __name__ == "__main__":
    main()
