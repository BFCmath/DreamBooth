#!/usr/bin/env python3
"""
DreamBooth + ControlNet Fine-tuning Script for Stable Diffusion 1.5
Fine-tunes a PRETRAINED ControlNet with DreamBooth identity learning.

This script enables fine-tuning an existing ControlNet (e.g., OpenPose) to be 
optimized for a specific identity using DreamBooth technique.

Key concepts:
- Uses a PRETRAINED ControlNet (e.g., lllyasviel/control_v11p_sd15_openpose)
- UNet is FROZEN - only ControlNet is trained
- DreamBooth rare token (sks) binds identity to the ControlNet
- Prior preservation prevents catastrophic forgetting

Based on:
- diffusers ControlNet training: https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py
- DreamBooth paper: https://arxiv.org/abs/2208.12242

Dataset Structure Required:
    data/
    ├── instance_images/      # Instance images (the identity you want to learn)
    │   ├── 001.png
    │   └── ...
    ├── conditioning/         # Conditioning images (pose/edges) for each instance
    │   ├── 001.png           # Must match instance image names
    │   └── ...
    └── prompts.txt           # Optional: One prompt per line

Key Hyperparameters for Fine-tuning:
- CONTROLNET_MODEL: Pretrained ControlNet to fine-tune (required)
- instance_prompt: "a photo of sks cat" (sks is the rare token identifier)
- class_prompt: "a photo of cat" (for prior preservation)
- learning_rate: 1e-6 (very low for fine-tuning)
- max_train_steps: 200-400 (small dataset, pretrained model)
- repeats: 10-20 (repeat instance images)

Environment Variables:
    CONTROLNET_MODEL: Path or HuggingFace ID of pretrained ControlNet
                     Default: lllyasviel/control_v11p_sd15_openpose
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
    Dataset for DreamBooth-style ControlNet training.
    
    Combines:
    - Instance images (the specific identity to learn) with conditioning
    - Class images (for prior preservation) with conditioning
    
    Expects:
    - instance_images_dir: directory with instance images (001.png, 002.png, ...)
    - conditioning_dir: directory with conditioning images (same names as instance)
    - instance_prompt: prompt with rare identifier (e.g., "a photo of sks person")
    
    Optional for prior preservation:
    - class_images_dir: directory with class images
    - class_conditioning_dir: directory with class conditioning images
    - class_prompt: prompt for class images (e.g., "a photo of person")
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
        
        # Load optional prompts file (overrides instance_prompt if provided)
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
        
        # Get prompt (from file or default instance_prompt)
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
            
            # MUST use class_conditioning_dir for proper (image, conditioning) pairs
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
# Class Image Generation (for Prior Preservation)
# =============================================================================
def generate_class_images(
    class_images_dir: str,
    class_conditioning_dir: str,  # NEW: also save conditioning
    class_prompt: str,
    num_class_images: int,
    pretrained_model: str,
    controlnet_path: str = None,
    device: str = "cuda",
    sample_batch_size: int = 4,  # Generate 4 images at a time for speed
):
    """
    Generate class images AND their conditioning (Canny edges) for prior preservation.
    
    This ensures proper (image, conditioning) pairs for the prior loss calculation.
    Without matching conditioning, the prior loss would use mismatched pairs.
    
    Args:
        class_images_dir: Directory to save generated class images
        class_conditioning_dir: Directory to save Canny edges for class images
        sample_batch_size: Number of images to generate per batch
    """
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
        # Also check if conditioning exists
        existing_cond = list(cond_dir.glob("*.png")) + list(cond_dir.glob("*.jpg"))
        if len(existing_cond) >= num_class_images:
            print(f"✅ Found {cur_class_images} class images + {len(existing_cond)} conditioning (needed {num_class_images})")
            return
    
    num_to_generate = num_class_images - cur_class_images
    print(f"📸 Generating {num_to_generate} class images with prompt: '{class_prompt}'")
    print(f"   Also extracting Canny edges for each class image")
    print(f"   Batch size: {sample_batch_size} (adjust SAMPLE_BATCH_SIZE to use more GPU)")
    
    from diffusers import StableDiffusionPipeline
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Generate in batches for speed
    num_batches = (num_to_generate + sample_batch_size - 1) // sample_batch_size
    generated = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Generating class images + conditioning"):
        # Calculate how many images to generate in this batch
        remaining = num_to_generate - generated
        batch_count = min(sample_batch_size, remaining)
        
        # Generate batch
        images = pipeline(
            [class_prompt] * batch_count,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images
        
        # Save each image AND extract Canny edges
        for i, image in enumerate(images):
            idx = cur_class_images + generated + i
            
            # Save class image
            image_path = class_dir / f"class_{idx:04d}.png"
            image.save(image_path)
            
            # Extract and save Canny edges (conditioning)
            image_np = np.array(image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            cond_path = cond_dir / f"class_{idx:04d}.png"
            Image.fromarray(edges_rgb).save(cond_path)
        
        generated += batch_count
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ Generated {num_to_generate} class images + conditioning")


# =============================================================================
# Training Function
# =============================================================================
def train(
    data_dir: str = "./data",
    output_dir: str = "./output/controlnet-dreambooth",
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    # DreamBooth-specific parameters
    instance_prompt: str = "a photo of sks person",
    class_prompt: str = "a photo of person",
    with_prior_preservation: bool = True,
    prior_loss_weight: float = 1.0,
    num_class_images: int = 100,
    sample_batch_size: int = 4,  # Batch size for generating class images
    # Training parameters
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-6,  # Lower LR for fine-tuning pretrained ControlNet
    max_train_steps: int = 400,
    checkpointing_steps: int = 200,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = True,
    seed: int = 42,
    repeats: int = 20,  # Repeat instance images
):
    """
    Train a ControlNet model with DreamBooth technique for identity-oriented generation.
    
    This combines:
    1. ControlNet conditioning (pose, edges, etc.)
    2. DreamBooth identity learning (rare token + prior preservation)
    
    Args:
        data_dir: Directory containing instance_images/, conditioning/, etc.
        output_dir: Where to save the trained ControlNet
        pretrained_model: Base SD model to use
        
        # DreamBooth parameters
        instance_prompt: Prompt with rare identifier (e.g., "a photo of sks person")
        class_prompt: General class prompt (e.g., "a photo of person")
        with_prior_preservation: Enable prior preservation loss
        prior_loss_weight: Weight of prior preservation loss (default 1.0)
        num_class_images: Number of class images for prior preservation
        
        # Training parameters
        resolution: Training resolution
        train_batch_size: Batch size per device
        gradient_accumulation_steps: Accumulate gradients for effective larger batch
        learning_rate: Learning rate (lower for DreamBooth, e.g., 5e-6)
        max_train_steps: Total training steps
        checkpointing_steps: Save checkpoint every N steps
        mixed_precision: Use FP16 for lower memory
        gradient_checkpointing: Trade compute for memory
        use_8bit_adam: Use 8-bit Adam for lower memory
        seed: Random seed
        repeats: Times to repeat instance images in dataset
    """
    
    print("=" * 60)
    print("DreamBooth + ControlNet Fine-tuning")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {pretrained_model}")
    print()
    print("DreamBooth Configuration:")
    print(f"  Instance prompt: {instance_prompt}")
    print(f"  Class prompt: {class_prompt}")
    print(f"  Prior preservation: {with_prior_preservation}")
    print(f"  Prior loss weight: {prior_loss_weight}")
    print(f"  Num class images: {num_class_images}")
    print()
    print("Training Configuration:")
    print(f"  Resolution: {resolution}")
    print(f"  Batch size: {train_batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {train_batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max steps: {max_train_steps}")
    print(f"  Mixed precision: {mixed_precision}")
    print(f"  Gradient checkpointing: {gradient_checkpointing}")
    print(f"  8-bit Adam: {use_8bit_adam}")
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
    # Initialize ControlNet
    # =========================================
    print()
    print("=" * 60)
    print("Phase 3: Initializing ControlNet")
    print("=" * 60)
    
    # Load pretrained ControlNet (required for fine-tuning)
    controlnet_model = os.environ.get("CONTROLNET_MODEL", "lllyasviel/control_v11p_sd15_openpose")
    
    print(f"   📂 Loading pretrained ControlNet: {controlnet_model}")
    controlnet = ControlNetModel.from_pretrained(controlnet_model)
    print("   ✅ ControlNet loaded (fine-tuning mode)")
    print("   ℹ️  UNet is FROZEN - only ControlNet will be trained")
    
    controlnet.to(device)  # Keep in float32 for training stability
    
    if gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        print("   ✅ Gradient checkpointing enabled")
    
    print_vram("After ControlNet init")
    
    # =========================================
    # Dataset
    # =========================================
    print()
    print("=" * 60)
    print("Phase 4: Loading dataset")
    print("=" * 60)
    
    # Check for prompts file
    prompts_file = data_path / "prompts.txt"
    prompts_file = str(prompts_file) if prompts_file.exists() else None
    
    # Class images and conditioning directories (created by generate_class_images if prior preservation)
    class_images_dir = data_path / "class_images" if with_prior_preservation else None
    # class_conditioning_dir is created alongside class_images by generate_class_images
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
    # Optimizer
    # =========================================
    print()
    print("=" * 60)
    print("Phase 5: Setting up optimizer")
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
    print("🚀 Phase 6: Starting Training")
    print("=" * 60)
    print(f"  Prior preservation: {with_prior_preservation}")
    print(f"  Prior loss weight: {prior_loss_weight}")
    print()
    
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
            
            # Get target (noise for epsilon prediction)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Calculate loss with prior preservation
            if with_prior_preservation:
                # Split predictions into instance and class
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                
                # Instance loss (identity-specific)
                instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Prior loss (preserve general knowledge)
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                
                # Combined loss
                loss = instance_loss + prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
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
    print("To use your trained DreamBooth ControlNet:")
    print("```python")
    print("from diffusers import StableDiffusionControlNetPipeline, ControlNetModel")
    print(f'controlnet = ControlNetModel.from_pretrained("{output_dir}")')
    print(f'pipe = StableDiffusionControlNetPipeline.from_pretrained("{pretrained_model}", controlnet=controlnet)')
    print(f'image = pipe("{instance_prompt}", image=pose_image).images[0]')
    print("```")


def main():
    parser = argparse.ArgumentParser(description="DreamBooth + ControlNet Fine-tuning")
    
    # Data and output
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output/controlnet-dreambooth")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    
    # DreamBooth parameters
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person",
                       help="Prompt with rare identifier for the instance")
    parser.add_argument("--class_prompt", type=str, default="a photo of person",
                       help="General class prompt for prior preservation")
    parser.add_argument("--with_prior_preservation", action="store_true",
                       help="Enable prior preservation loss")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0,
                       help="Weight of prior preservation loss")
    parser.add_argument("--num_class_images", type=int, default=100,
                       help="Number of class images for prior preservation")
    parser.add_argument("--sample_batch_size", type=int, default=4,
                       help="Batch size for class image generation (increase to use more GPU)")
    
    # Training parameters
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                       help="Learning rate (very low for fine-tuning pretrained ControlNet)")
    parser.add_argument("--max_train_steps", type=int, default=400)
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
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        with_prior_preservation=args.with_prior_preservation,
        prior_loss_weight=args.prior_loss_weight,
        num_class_images=args.num_class_images,
        sample_batch_size=args.sample_batch_size,
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
