# scripts/train_stage2.py
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
    unet.config.in_channels = 8
    
    return unet

class Stage2Trainer:
    """Fine-tune InstructPix2Pix on synthetic fine-grained makeup data"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize accelerator with fp16 mixed precision for speed
        self.accelerator = Accelerator(
            mixed_precision=config['mixed_precision'],
            gradient_accumulation_steps=config['gradient_accumulation_steps']
        )
        
        # Check if resuming from Stage 2 checkpoint
        if config.get('resume_from_checkpoint'):
            print(f"Resuming Stage 2 from checkpoint: {config['resume_from_checkpoint']}")
            self.load_stage2_checkpoint(config['resume_from_checkpoint'])
        else:
            print(f"Loading Stage 1 model from {config['base_model']}...")
            self.load_stage1_model(config['base_model'])
        
        print("Model loaded successfully!")
        
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
        
        # Learning rate scheduler with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Dataset - Stage 2 uses synthetic pairs
        self.train_dataset = InstructPix2PixDataset(
            config['data_dir'],
            resolution=config['resolution'],
            is_stage1=False  # Use stage 2 data format
        )
        
        print(f"Training dataset size: {len(self.train_dataset)}")
        
        # Create validation split (10%)
        val_size = max(1, int(len(self.train_dataset) * 0.1))
        train_size = len(self.train_dataset) - val_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train size: {len(self.train_dataset)}, Validation size: {len(self.val_dataset)}")
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config['train_batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config['train_batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # Prepare with accelerator
        self.unet, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )
        
        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device)
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"GPU Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def load_stage1_model(self, base_model_path):
        """Load Stage 1 trained model"""
        if os.path.exists(os.path.join(base_model_path, 'unet')):
            # Loading from checkpoint
            print("Loading from checkpoint structure...")
            
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
            print(f"Loading trained weights from {base_model_path}/unet...")
            checkpoint_unet_path = os.path.join(base_model_path, "unet")
            
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
        else:
            # Loading from full pipeline
            print("Loading from full pipeline...")
            self.vae = AutoencoderKL.from_pretrained(
                base_model_path,
                subfolder="vae",
                torch_dtype=torch.float16
            )
            
            self.tokenizer = CLIPTokenizer.from_pretrained(
                base_model_path,
                subfolder="tokenizer"
            )
            
            self.text_encoder = CLIPTextModel.from_pretrained(
                base_model_path,
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
            
            # Load trained weights from full pipeline
            checkpoint_unet_path = os.path.join(base_model_path, "unet")
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
            
            self.unet.load_state_dict(state_dict, strict=True)
            
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                base_model_path,
                subfolder="scheduler"
            )
    
    def load_stage2_checkpoint(self, checkpoint_path):
        """Load Stage 2 checkpoint for resuming training"""
        print(f"Loading Stage 2 checkpoint from {checkpoint_path}...")
        
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
        
        # Load the trained weights from Stage 2 checkpoint
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
        print("Stage 2 checkpoint loaded successfully!")
        
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
    
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
    
    def validate(self):
        """Run validation"""
        self.unet.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                original_images = batch['original_image'].to(
                    self.accelerator.device, 
                    dtype=torch.float16
                )
                edited_images = batch['edited_image'].to(
                    self.accelerator.device, 
                    dtype=torch.float16
                )
                
                # Encode images
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
                ).long()
                
                # Add noise
                noisy_latents = self.scheduler.add_noise(latents_edited, noise, timesteps)
                
                # Concatenate
                latent_model_input = torch.cat([noisy_latents, latents_orig], dim=1)
                
                # Predict
                model_pred = self.unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                val_loss += loss.item()
        
        self.unet.train()
        return val_loss / len(self.val_dataloader)
    
    def train(self, num_epochs, start_epoch=0):
        """Main training loop"""
        global_step = start_epoch * len(self.train_dataloader)
        best_val_loss = float('inf')
        
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
                    
                    # Concatenate with original latents
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
            
            # Validation
            print("\nRunning validation...")
            val_loss = self.validate()
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"\nEpoch {epoch} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f} - Saving best model...")
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
        'base_model': 'models/stage1_model',  # Load from Stage 1 output
        'resume_from_checkpoint': None,  # Set to 'models/stage2_model/checkpoint-epoch_X' to resume
        'data_dir': 'data/stage2_synthetic',
        'output_dir': 'models/stage2_model',
        'resolution': 512,  # Full resolution
        'train_batch_size': 8,  # Larger batch size for 30GB GPU
        'num_workers': 4,  # More workers for faster data loading
        'learning_rate': 5e-6,  # Lower LR for fine-tuning (half of stage 1)
        'gradient_accumulation_steps': 2,
        'mixed_precision': 'fp16',  # Enable fp16 for speed
        'checkpointing_steps': 500,
        'num_epochs': 50,  # Fewer epochs for fine-tuning
        'start_epoch': 0
    }
    
    print("="*70)
    if config.get('resume_from_checkpoint'):
        print("InstructPix2Pix Training - Stage 2 (RESUMING - 30GB GPU)")
    else:
        print("InstructPix2Pix Training - Stage 2 (30GB GPU Optimized)")
    print("="*70)
    if config.get('resume_from_checkpoint'):
        print(f"Resuming from: {config['resume_from_checkpoint']}")
        print(f"Starting epoch: {config['start_epoch']}")
    print(f"Base model: {config['base_model']}")
    print(f"Target epochs: {config['num_epochs']}")
    print(f"Resolution: {config['resolution']}x{config['resolution']}")
    print(f"Batch size: {config['train_batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Mixed precision: {config['mixed_precision']}")
    print(f"Num workers: {config['num_workers']}")
    print("="*70)
    
    trainer = Stage2Trainer(config)
    
    # Load optimizer state if resuming
    if config.get('resume_from_checkpoint'):
        trainer.load_optimizer_state(config['resume_from_checkpoint'])
    
    trainer.train(num_epochs=config['num_epochs'], start_epoch=config['start_epoch'])
    
    print("\n" + "="*70)
    print("Stage 2 Training completed!")
    print(f"Model saved to: {config['output_dir']}")
    print("="*70)

if __name__ == "__main__":
    main()
