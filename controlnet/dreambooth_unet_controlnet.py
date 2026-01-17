#!/usr/bin/env python3
"""
DreamBooth + ControlNet Training Script (UNet Training Mode)

This script trains the UNet for identity learning while using a FROZEN pretrained
ControlNet for structural conditioning. This is the CORRECT approach for 
identity-oriented fine-tuning with ControlNet.

Key difference from dreambooth_controlnet.py:
- UNet is TRAINED (identity learning happens here)
- ControlNet is FROZEN (only provides structural conditioning)

This approach:
1. Preserves the base ControlNet's conditioning capabilities
2. Allows UNet to learn the specific identity (sks token binding)
3. Uses prior preservation to prevent catastrophic forgetting

Based on:
- DreamBooth paper: https://arxiv.org/abs/2208.12242
- ControlNet paper: https://arxiv.org/abs/2302.05543

Dataset Structure Required:
    data/
    ├── instance_images/      # Instance images (the identity you want to learn)
    │   ├── 001.png
    │   └── ...
    ├── conditioning/         # Conditioning images (pose/edges) for each instance
    │   ├── 001.png           # Must match instance image names
    │   └── ...
    └── prompts.txt           # Optional: One prompt per line

Key Hyperparameters:
- instance_prompt: "a photo of sks cat" (sks is the rare token identifier)
- class_prompt: "a photo of cat" (for prior preservation)
- learning_rate: 5e-6 (standard DreamBooth LR)
- max_train_steps: 400-800 (typical for DreamBooth)
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
    StableDiffusionControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer


# =============================================================================
# Dataset for DreamBooth + ControlNet
# =============================================================================
class DreamBoothControlNetDataset(Dataset):
    """
    Dataset for DreamBooth-style training with ControlNet conditioning.
    
    Combines:
    - Instance images (the specific identity to learn) with conditioning
    - Class images (for prior preservation) with conditioning
    """
    
    def __init__(
        self,
        instance_images_dir: str,
        conditioning_dir: str,
        instance_prompt: str,
        tokenizer,
        class_images_dir: str = None,
        class_conditioning_dir: str = None,
        class_prompt: str = None,
        prompts_file: str = None,
        resolution: int = 512,
        repeats: int = 1,
    ):
        self.instance_images_dir = Path(instance_images_dir)
        self.conditioning_dir = Path(conditioning_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Load optional prompts file
        self.prompts = None
        if prompts_file and os.path.exists(prompts_file):
            with open(prompts_file, "r") as f:
                self.prompts = [line.strip() for line in f.readlines() if line.strip()]
            print(f"📝 Loaded {len(self.prompts)} prompts from file")
        
        # Find instance image files
        self.instance_images = sorted([
            f for f in self.instance_images_dir.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        ])
        self.num_instance_images = len(self.instance_images)
        
        if self.num_instance_images == 0:
            raise ValueError(f"No images found in {instance_images_dir}")
        
        print(f"📊 Found {self.num_instance_images} instance images")
        
        # Class images (for prior preservation)
        self.class_images = []
        self.class_conditioning_dir = None
        self.class_prompt = class_prompt
        
        if class_images_dir and os.path.exists(class_images_dir):
            self.class_images_dir = Path(class_images_dir)
            self.class_images = sorted([
                f for f in self.class_images_dir.iterdir()
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
            ])
            self.num_class_images = len(self.class_images)
            print(f"📊 Found {self.num_class_images} class images for prior preservation")
            
            if class_conditioning_dir and os.path.exists(class_conditioning_dir):
                self.class_conditioning_dir = Path(class_conditioning_dir)
                print(f"📊 Using class conditioning from {class_conditioning_dir}")
        else:
            self.num_class_images = 0
        
        # Calculate dataset length
        self._length = self.num_instance_images * repeats
        if self.num_class_images > 0:
            self._length = max(self._length, self.num_class_images)
        
        print(f"📊 Total dataset length: {self._length}")
        
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
        return self._length
    
    def _load_image_and_conditioning(self, image_path, cond_dir):
        """Load an image and its corresponding conditioning image."""
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_transforms(image)
        
        # Find conditioning image
        cond_path = cond_dir / image_path.name
        if not cond_path.exists():
            cond_files = list(cond_dir.glob(f"{image_path.stem}.*"))
            if cond_files:
                cond_path = cond_files[0]
            else:
                raise FileNotFoundError(f"No conditioning image for {image_path.name} in {cond_dir}")
        
        cond_image = Image.open(cond_path).convert("RGB")
        conditioning_pixel_values = self.conditioning_transforms(cond_image)
        
        return pixel_values, conditioning_pixel_values
    
    def __getitem__(self, idx):
        example = {}
        
        # Instance image
        instance_idx = idx % self.num_instance_images
        instance_path = self.instance_images[instance_idx]
        
        pixel_values, conditioning_pixel_values = self._load_image_and_conditioning(
            instance_path, self.conditioning_dir
        )
        
        example["instance_pixel_values"] = pixel_values
        example["instance_conditioning"] = conditioning_pixel_values
        
        # Get prompt
        if self.prompts is not None and instance_idx < len(self.prompts):
            prompt = self.prompts[instance_idx]
        else:
            prompt = self.instance_prompt
        
        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        # Class image (for prior preservation)
        if self.num_class_images > 0:
            class_idx = idx % self.num_class_images
            class_path = self.class_images[class_idx]
            
            if self.class_conditioning_dir is None:
                raise ValueError(
                    "Prior preservation requires class_conditioning_dir! "
                    "Run training with --with_prior_preservation to auto-generate class images + conditioning."
                )
            
            class_pixels, class_cond = self._load_image_and_conditioning(
                class_path, self.class_conditioning_dir
            )
            example["class_pixel_values"] = class_pixels
            example["class_conditioning"] = class_cond
            
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
        
        return example


def collate_fn(examples, with_prior_preservation=False):
    """Collate function for DataLoader."""
    instance_pixel_values = [ex["instance_pixel_values"] for ex in examples]
    instance_conditioning = [ex["instance_conditioning"] for ex in examples]
    instance_prompt_ids = [ex["instance_prompt_ids"] for ex in examples]
    
    if with_prior_preservation:
        class_pixel_values = [ex["class_pixel_values"] for ex in examples]
        class_conditioning = [ex["class_conditioning"] for ex in examples]
        class_prompt_ids = [ex["class_prompt_ids"] for ex in examples]
        
        pixel_values = torch.stack(instance_pixel_values + class_pixel_values)
        conditioning = torch.stack(instance_conditioning + class_conditioning)
        input_ids = torch.stack(instance_prompt_ids + class_prompt_ids)
    else:
        pixel_values = torch.stack(instance_pixel_values)
        conditioning = torch.stack(instance_conditioning)
        input_ids = torch.stack(instance_prompt_ids)
    
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning,
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
# Conditioning Extraction Utilities
# =============================================================================
def extract_conditioning(image_np, conditioning_type: str):
    """
    Extract conditioning from an image based on the specified type.
    """
    import cv2
    import numpy as np
    
    if conditioning_type == "canny":
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif conditioning_type == "hed":
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif conditioning_type == "none":
        return None
    
    else:
        raise ValueError(f"Unknown conditioning_type: {conditioning_type}")


# =============================================================================
# Class Image Generation (for Prior Preservation)
# =============================================================================
def generate_class_images(
    class_images_dir: str,
    class_conditioning_dir: str,
    class_prompt: str,
    num_class_images: int,
    pretrained_model: str,
    conditioning_type: str = "canny",
    device: str = "cuda",
    sample_batch_size: int = 4,
):
    """Generate class images AND their conditioning for prior preservation."""
    import cv2
    import numpy as np
    
    class_dir = Path(class_images_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    
    cond_dir = Path(class_conditioning_dir)
    cond_dir.mkdir(parents=True, exist_ok=True)
    
    # Count existing images
    existing_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    cur_class_images = len(existing_images)
    
    if cur_class_images >= num_class_images:
        existing_cond = list(cond_dir.glob("*.png")) + list(cond_dir.glob("*.jpg"))
        if len(existing_cond) >= num_class_images:
            print(f"✅ Found {cur_class_images} class images + {len(existing_cond)} conditioning")
            return
    
    num_to_generate = num_class_images - cur_class_images
    print(f"📸 Generating {num_to_generate} class images with prompt: '{class_prompt}'")
    if conditioning_type != "none":
        print(f"   Also extracting {conditioning_type} conditioning for each class image")
    
    from diffusers import StableDiffusionPipeline
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    num_batches = (num_to_generate + sample_batch_size - 1) // sample_batch_size
    generated = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Generating class images"):
        remaining = num_to_generate - generated
        batch_count = min(sample_batch_size, remaining)
        
        images = pipeline(
            [class_prompt] * batch_count,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images
        
        for i, image in enumerate(images):
            idx = cur_class_images + generated + i
            
            image_path = class_dir / f"class_{idx:04d}.png"
            image.save(image_path)
            
            if conditioning_type != "none":
                image_np = np.array(image)
                cond_rgb = extract_conditioning(image_np, conditioning_type)
                cond_path = cond_dir / f"class_{idx:04d}.png"
                Image.fromarray(cond_rgb).save(cond_path)
        
        generated += batch_count
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ Generated {num_to_generate} class images + conditioning")


# =============================================================================
# Training Function - TRAINS UNET (not ControlNet)
# =============================================================================
def train(
    data_dir: str = "./data",
    output_dir: str = "./output/dreambooth-unet-controlnet",
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model: str = "lllyasviel/control_v11p_sd15_canny",
    # DreamBooth-specific parameters
    instance_prompt: str = "a photo of sks cat",
    class_prompt: str = "a photo of cat",
    with_prior_preservation: bool = True,
    prior_loss_weight: float = 1.0,
    num_class_images: int = 100,
    sample_batch_size: int = 4,
    conditioning_type: str = "canny",
    # Training parameters
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,  # Standard DreamBooth LR for UNet
    max_train_steps: int = 800,
    checkpointing_steps: int = 200,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = True,
    seed: int = 42,
    repeats: int = 20,
):
    """
    Train UNet with DreamBooth technique using frozen ControlNet for conditioning.
    
    This is the CORRECT approach for identity-oriented fine-tuning:
    - UNet learns the identity (sks token binding)
    - ControlNet provides structural conditioning (frozen)
    """
    
    print("=" * 60)
    print("DreamBooth + ControlNet (UNet Training Mode)")
    print("=" * 60)
    print()
    print("🔑 KEY DIFFERENCE: This trains UNET for identity learning!")
    print("   ControlNet is FROZEN (only provides structural guidance)")
    print()
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {pretrained_model}")
    print(f"ControlNet: {controlnet_model}")
    print()
    print("DreamBooth Configuration:")
    print(f"  Instance prompt: {instance_prompt}")
    print(f"  Class prompt: {class_prompt}")
    print(f"  Prior preservation: {with_prior_preservation}")
    print(f"  Prior loss weight: {prior_loss_weight}")
    print(f"  Num class images: {num_class_images}")
    print(f"  Conditioning type: {conditioning_type}")
    print()
    print("Training Configuration:")
    print(f"  Resolution: {resolution}")
    print(f"  Batch size: {train_batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max steps: {max_train_steps}")
    print(f"  Instance repeats: {repeats}")
    print("=" * 60)
    print()
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if mixed_precision and device == "cuda" else torch.float32
    
    if seed is not None:
        torch.manual_seed(seed)
    
    data_path = Path(data_dir)
    
    print_vram("Initial")
    
    # =========================================
    # Generate Class Images (if needed)
    # =========================================
    if with_prior_preservation:
        print("=" * 60)
        print("Phase 1: Generating class images for prior preservation")
        print("=" * 60)
        
        class_images_dir = data_path / "class_images"
        class_conditioning_dir = data_path / "class_conditioning"
        generate_class_images(
            class_images_dir=str(class_images_dir),
            class_conditioning_dir=str(class_conditioning_dir),
            class_prompt=class_prompt,
            num_class_images=num_class_images,
            pretrained_model=pretrained_model,
            conditioning_type=conditioning_type,
            device=device,
            sample_batch_size=sample_batch_size,
        )
        print()
    
    # =========================================
    # Load Models
    # =========================================
    print("=" * 60)
    print("Phase 2: Loading models")
    print("=" * 60)
    
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
    
    # UNet (TRAINABLE - this is where identity learning happens!)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    unet.to(device)  # Keep in float32 for training stability
    print("   ✅ UNet loaded (TRAINABLE - identity learning)")
    
    # ControlNet (FROZEN - only provides structural conditioning)
    controlnet = ControlNetModel.from_pretrained(controlnet_model)
    controlnet.requires_grad_(False)
    controlnet.to(device, dtype=weight_dtype)
    print(f"   ✅ ControlNet loaded from {controlnet_model} (FROZEN)")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    print("   ✅ Noise scheduler loaded")
    
    print_vram("After loading models")
    
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("   ✅ UNet gradient checkpointing enabled")
    
    # =========================================
    # Dataset
    # =========================================
    print()
    print("=" * 60)
    print("Phase 3: Loading dataset")
    print("=" * 60)
    
    prompts_file = data_path / "prompts.txt"
    prompts_file = str(prompts_file) if prompts_file.exists() else None
    
    class_images_dir = data_path / "class_images" if with_prior_preservation else None
    class_conditioning_dir = str(data_path / "class_conditioning") if with_prior_preservation else None
    
    dataset = DreamBoothControlNetDataset(
        instance_images_dir=str(data_path / "instance_images"),
        conditioning_dir=str(data_path / "conditioning"),
        instance_prompt=instance_prompt,
        tokenizer=tokenizer,
        class_images_dir=str(class_images_dir) if class_images_dir else None,
        class_conditioning_dir=class_conditioning_dir,
        class_prompt=class_prompt if with_prior_preservation else None,
        prompts_file=prompts_file,
        resolution=resolution,
        repeats=repeats,
    )
    
    def collate_fn_wrapper(examples):
        return collate_fn(examples, with_prior_preservation=with_prior_preservation)
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_wrapper,
    )
    print(f"   ✅ Dataloader created: {len(dataloader)} batches")
    
    # =========================================
    # Optimizer (for UNet only)
    # =========================================
    print()
    print("=" * 60)
    print("Phase 4: Setting up optimizer (for UNet)")
    print("=" * 60)
    
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
        unet.parameters(),  # Only UNet parameters!
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
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
    print("🚀 Phase 5: Starting Training (UNet)")
    print("=" * 60)
    print(f"  Training: UNet (identity learning)")
    print(f"  Frozen: ControlNet, VAE, Text Encoder")
    print(f"  Prior preservation: {with_prior_preservation}")
    print()
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Training UNet")
    
    unet.train()
    controlnet.eval()  # ControlNet stays in eval mode
    
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
            
            # Get ControlNet output (frozen, no gradients)
            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_pixel_values,
                    return_dict=False,
                )
            
            # Predict noise with UNet (TRAINABLE) using ControlNet guidance
            model_pred = unet(
                noisy_latents.float(),  # UNet in float32 for training
                timesteps,
                encoder_hidden_states=encoder_hidden_states.float(),
                down_block_additional_residuals=[s.float() for s in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.float(),
            ).sample
            
            # Get target
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Calculate loss with prior preservation
            if with_prior_preservation:
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                
                instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                
                loss = instance_loss + prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            loss = loss / gradient_accumulation_steps
            
            # Backward
            loss.backward()
            
            # Update weights
            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Logging
            progress_bar.update(1)
            effective_loss = loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix(loss=effective_loss)
            
            global_step += 1
            
            # Periodic logging
            if global_step % 50 == 0:
                log_msg = f"\n📊 Step {global_step}/{max_train_steps} | Loss: {effective_loss:.4f}"
                if with_prior_preservation:
                    log_msg += f" (instance: {instance_loss.item():.4f}, prior: {prior_loss.item():.4f})"
                print(log_msg)
                print_vram("Training")
            
            # Save checkpoint
            if global_step % checkpointing_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Save only the UNet (what we trained)
                unet_to_save = unet
                unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet"))
                print(f"\n💾 UNet checkpoint saved: {checkpoint_dir}")
            
            if global_step >= max_train_steps:
                break
    
    # =========================================
    # Save Final Model
    # =========================================
    print()
    print("=" * 60)
    print("💾 Saving Final Model")
    print("=" * 60)
    
    # Save the full pipeline with trained UNet
    from diffusers import StableDiffusionControlNetPipeline
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        unet=unet,
        controlnet=controlnet,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        safety_checker=None,
    )
    pipeline.save_pretrained(output_dir)
    print(f"✅ Full pipeline saved to: {output_dir}")
    
    # Also save just the UNet separately for easy loading
    unet.save_pretrained(os.path.join(output_dir, "unet_trained"))
    print(f"✅ Trained UNet also saved to: {output_dir}/unet_trained")
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print()
    print("To use your trained model:")
    print("```python")
    print("from diffusers import StableDiffusionControlNetPipeline, ControlNetModel")
    print(f'pipe = StableDiffusionControlNetPipeline.from_pretrained("{output_dir}")')
    print(f'image = pipe("{instance_prompt}", image=conditioning_image).images[0]')
    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="DreamBooth + ControlNet (UNet Training Mode) - trains UNet for identity learning"
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output/dreambooth-unet-controlnet")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11p_sd15_canny",
                       help="Pretrained ControlNet to use (frozen)")
    
    # DreamBooth parameters
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks cat",
                       help="Prompt with rare identifier for the instance")
    parser.add_argument("--class_prompt", type=str, default="a photo of cat",
                       help="General class prompt for prior preservation")
    parser.add_argument("--with_prior_preservation", action="store_true",
                       help="Enable prior preservation loss")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0,
                       help="Weight of prior preservation loss")
    parser.add_argument("--num_class_images", type=int, default=100,
                       help="Number of class images for prior preservation")
    parser.add_argument("--sample_batch_size", type=int, default=4,
                       help="Batch size for class image generation")
    parser.add_argument("--conditioning_type", type=str, default="canny",
                       choices=["canny", "hed", "none"],
                       help="Type of conditioning to extract for class images")
    
    # Training parameters
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate (standard DreamBooth LR for UNet)")
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20,
                       help="Times to repeat instance images")
    
    # Optimization flags
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_8bit_adam", action="store_true")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        controlnet_model=args.controlnet_model,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        with_prior_preservation=args.with_prior_preservation,
        prior_loss_weight=args.prior_loss_weight,
        num_class_images=args.num_class_images,
        sample_batch_size=args.sample_batch_size,
        conditioning_type=args.conditioning_type,
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
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
