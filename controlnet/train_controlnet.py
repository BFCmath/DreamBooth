#!/usr/bin/env python3
"""
ControlNet Fine-tuning Script for Stable Diffusion 1.5
Simplified version for testing fine-tuning on Kaggle P100

Based on: https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py

Dataset Structure Required:
    data/
    ├── images/           # Ground truth images (what we want to generate)
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    ├── conditioning/     # Conditioning images (pose/edges/etc)
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    └── prompts.txt       # One prompt per line, matching image order
"""

import argparse
import os
import math
from pathlib import Path
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    ControlNetModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer


# =============================================================================
# Dataset
# =============================================================================
class ControlNetDataset(Dataset):
    """
    Simple dataset for ControlNet training.
    
    Expects:
    - images_dir: directory with ground truth images (001.png, 002.png, ...)
    - conditioning_dir: directory with conditioning images (same names)
    - prompts_file: text file with one prompt per line
    """
    
    def __init__(
        self,
        images_dir: str,
        conditioning_dir: str,
        prompts_file: str,
        tokenizer,
        resolution: int = 512,
    ):
        self.images_dir = Path(images_dir)
        self.conditioning_dir = Path(conditioning_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Load prompts
        with open(prompts_file, "r") as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Find image files
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        
        # Ensure counts match
        if len(self.image_files) != len(self.prompts):
            print(f"⚠️  Warning: {len(self.image_files)} images but {len(self.prompts)} prompts")
            min_count = min(len(self.image_files), len(self.prompts))
            self.image_files = self.image_files[:min_count]
            self.prompts = self.prompts[:min_count]
        
        print(f"📊 Dataset: {len(self.image_files)} samples")
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.conditioning_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load ground truth image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_transforms(image)
        
        # Load conditioning image (same filename in conditioning dir)
        cond_path = self.conditioning_dir / image_path.name
        if not cond_path.exists():
            # Try to find any matching file
            cond_files = list(self.conditioning_dir.glob(f"{image_path.stem}.*"))
            if cond_files:
                cond_path = cond_files[0]
            else:
                raise FileNotFoundError(f"No conditioning image for {image_path.name}")
        
        cond_image = Image.open(cond_path).convert("RGB")
        conditioning_pixel_values = self.conditioning_transforms(cond_image)
        
        # Tokenize prompt
        prompt = self.prompts[idx]
        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "input_ids": input_ids,
        }


# =============================================================================
# VRAM Debug Helper
# =============================================================================
def print_vram(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   📊 VRAM [{label}]: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


# =============================================================================
# Training Function
# =============================================================================
def train(
    data_dir: str = "./data",
    output_dir: str = "./output/controlnet-finetuned",
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    max_train_steps: int = 500,
    checkpointing_steps: int = 100,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = True,
    seed: int = 42,
):
    """
    Train/fine-tune a ControlNet model.
    
    Args:
        data_dir: Directory containing images/, conditioning/, prompts.txt
        output_dir: Where to save the trained ControlNet
        pretrained_model: Base SD model to use
        resolution: Training resolution
        train_batch_size: Batch size per device
        gradient_accumulation_steps: Accumulate gradients for effective larger batch
        learning_rate: Learning rate for optimizer
        max_train_steps: Total training steps
        checkpointing_steps: Save checkpoint every N steps
        mixed_precision: Use FP16 for lower memory
        gradient_checkpointing: Trade compute for memory
        use_8bit_adam: Use 8-bit Adam for lower memory
        seed: Random seed
    """
    
    print("=" * 60)
    print("ControlNet Fine-tuning")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {pretrained_model}")
    print(f"Resolution: {resolution}")
    print(f"Batch size: {train_batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {train_batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max steps: {max_train_steps}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"8-bit Adam: {use_8bit_adam}")
    print("=" * 60)
    print()
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if mixed_precision and device == "cuda" else torch.float32
    
    if seed is not None:
        torch.manual_seed(seed)
    
    print_vram("Initial")
    
    # =========================================
    # Load Models
    # =========================================
    print("⏳ Loading models...")
    
    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    print("   ✅ Tokenizer loaded")
    
    # Text encoder (frozen)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    print("   ✅ Text encoder loaded (frozen)")
    
    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    print("   ✅ VAE loaded (frozen)")
    
    # UNet (frozen, used for ControlNet output)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    print("   ✅ UNet loaded (frozen)")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    print("   ✅ Noise scheduler loaded")
    
    print_vram("After loading frozen models")
    
    # =========================================
    # Initialize ControlNet from UNet
    # =========================================
    print("⏳ Initializing ControlNet from UNet...")
    controlnet = ControlNetModel.from_unet(unet)
    controlnet.to(device)  # Keep in float32 for training stability
    print("   ✅ ControlNet initialized")
    
    if gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        print("   ✅ Gradient checkpointing enabled")
    
    print_vram("After ControlNet init")
    
    # =========================================
    # Dataset
    # =========================================
    print("⏳ Loading dataset...")
    data_path = Path(data_dir)
    
    dataset = ControlNetDataset(
        images_dir=str(data_path / "images"),
        conditioning_dir=str(data_path / "conditioning"),
        prompts_file=str(data_path / "prompts.txt"),
        tokenizer=tokenizer,
        resolution=resolution,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"   ✅ Dataloader created: {len(dataloader)} batches")
    
    # =========================================
    # Optimizer
    # =========================================
    print("⏳ Setting up optimizer...")
    
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("   ✅ Using 8-bit Adam optimizer")
        except ImportError:
            print("   ⚠️  bitsandbytes not found, using standard Adam")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        controlnet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # LR scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )
    print("   ✅ Optimizer and LR scheduler ready")
    
    # =========================================
    # Training Loop
    # =========================================
    print()
    print("=" * 60)
    print("🚀 Starting Training")
    print("=" * 60)
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Training")
    
    controlnet.train()
    
    while global_step < max_train_steps:
        for batch in dataloader:
            # Move to device
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            conditioning_pixel_values = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)
            
            # Encode images to latent space
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Get ControlNet output (ControlNet is in float32 for training stability)
            # Cast inputs to float32 for ControlNet
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents.float(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.float(),
                controlnet_cond=conditioning_pixel_values.float(),
                return_dict=False,
            )
            
            # Cast ControlNet outputs back to weight_dtype for UNet
            down_block_res_samples = [sample.to(dtype=weight_dtype) for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample.to(dtype=weight_dtype)
            
            # Predict noise with UNet using ControlNet guidance
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            # Calculate loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / gradient_accumulation_steps
            
            # Backward
            loss.backward()
            
            # Update weights
            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Logging
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps)
            
            global_step += 1
            
            # Periodic logging
            if global_step % 50 == 0:
                print(f"\n📊 Step {global_step}/{max_train_steps} | Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                print_vram("Training")
            
            # Save checkpoint
            if global_step % checkpointing_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                controlnet.save_pretrained(checkpoint_dir)
                print(f"\n💾 Checkpoint saved: {checkpoint_dir}")
            
            if global_step >= max_train_steps:
                break
    
    # =========================================
    # Save Final Model
    # =========================================
    print()
    print("=" * 60)
    print("💾 Saving Final Model")
    print("=" * 60)
    
    controlnet.save_pretrained(output_dir)
    print(f"✅ ControlNet saved to: {output_dir}")
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print()
    print("To use your trained ControlNet:")
    print("```python")
    print("from diffusers import StableDiffusionControlNetPipeline, ControlNetModel")
    print(f'controlnet = ControlNetModel.from_pretrained("{output_dir}")')
    print(f'pipe = StableDiffusionControlNetPipeline.from_pretrained("{pretrained_model}", controlnet=controlnet)')
    print("```")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ControlNet")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output/controlnet-finetuned")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_8bit_adam", action="store_true")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_8bit_adam=not args.no_8bit_adam,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
