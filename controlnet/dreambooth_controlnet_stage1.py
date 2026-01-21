#!/usr/bin/env python3
"""
Stage 1: Appearance Control Pretraining (No ControlNet)

This script trains the UNet + Text Encoder for identity learning WITHOUT ControlNet.
The model learns to associate the "sks" token with the specific identity appearance
purely from text embeddings.

Key differences from Stage 2:
- NO ControlNet (not loaded at all)
- Text Encoder is TRAINED by default (to learn sks token embedding)
- Task is pure reconstruction: text prompt → image

This approach:
1. Forces the UNet to rely entirely on text embeddings for identity
2. Trains the "sks" token to encode identity features
3. Prepares the model for Stage 2 (pose control)

After Stage 1, use Stage 2 to add pose control while preserving identity.

Based on:
- MagicPose paper: https://arxiv.org/pdf/2311.12052
- DreamBooth paper: https://arxiv.org/abs/2208.12242

Dataset Structure Required:
    data/
    ├── instance_images/      # Instance images (5-7 images of the identity)
    │   ├── 001.png
    │   └── ...
    └── prompts.txt           # Optional: One prompt per line

Key Hyperparameters:
- instance_prompt: "a sks humanoid robot" (sks is the rare token identifier)
- class_prompt: "a photo of person" (for prior preservation)
- learning_rate: 5e-6 (standard DreamBooth LR)
- max_train_steps: 800-1200 (more steps since no pose guidance)

Custom Diffusion Mode (--train_token_embedding):
- Trains only the "sks" token embedding (ID 48136) instead of full text encoder
- Trains only cross-attention K/V projections in UNet (not Q or out)
- Follows the Custom Diffusion paper approach for memory-efficient training
- Reference: https://arxiv.org/pdf/2212.04488v2
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
    StableDiffusionPipeline,  # Use standard pipeline for class generation (no ControlNet)
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model

# Local utilities
from utils import print_vram


# =============================================================================
# Dataset for Stage 1 (No Conditioning Required)
# =============================================================================
class Stage1Dataset(Dataset):
    """
    Dataset for Stage 1 appearance pretraining.
    
    Unlike Stage 2, this does NOT require conditioning images since
    we're training without ControlNet.
    """
    
    def __init__(
        self,
        instance_images_dir: str,
        instance_prompt: str,
        tokenizer,
        class_images_dir: str = None,
        class_prompt: str = None,
        prompts_file: str = None,
        resolution: int = 512,
        repeats: int = 1,
        augment_prompt_for_resize: bool = False,
    ):
        self.instance_images_dir = Path(instance_images_dir)
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.augment_prompt_for_resize = augment_prompt_for_resize
        
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
        self.class_prompt = class_prompt
        
        if class_images_dir and os.path.exists(class_images_dir):
            self.class_images_dir = Path(class_images_dir)
            self.class_images = sorted([
                f for f in self.class_images_dir.iterdir()
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
            ])
            self.num_class_images = len(self.class_images)
            print(f"📊 Found {self.num_class_images} class images for prior preservation")
        else:
            self.num_class_images = 0
        
        # Calculate dataset length
        self._length = self.num_instance_images * repeats
        if self.num_class_images > 0:
            self._length = max(self._length, self.num_class_images)
        
        print(f"📊 Total dataset length: {self._length}")
        
        # Custom Diffusion random scale range (0.4 to 1.4)
        self.scale_min = 0.4
        self.scale_max = 1.4
        self.pad_fill_color = (128, 128, 128)
    
    def __len__(self):
        return self._length
    
    def _apply_custom_diffusion_augmentation(self, image: Image.Image):
        """Apply Custom Diffusion random scale augmentation."""
        import random
        
        scale_factor = random.uniform(self.scale_min, self.scale_max)
        base_size = self.resolution
        
        if scale_factor < 1.0:
            new_size = int(base_size * scale_factor)
            resized_image = image.resize((new_size, new_size), Image.BILINEAR)
            padded_image = Image.new("RGB", (base_size, base_size), self.pad_fill_color)
            offset = (base_size - new_size) // 2
            padded_image.paste(resized_image, (offset, offset))
            return padded_image, scale_factor
        else:
            new_size = int(base_size * scale_factor)
            resized_image = image.resize((new_size, new_size), Image.BILINEAR)
            offset = (new_size - base_size) // 2
            cropped_image = resized_image.crop((offset, offset, offset + base_size, offset + base_size))
            return cropped_image, scale_factor
    
    def _load_image(self, image_path, apply_augmentation=True):
        """Load and preprocess an image."""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        scale_factor = 1.0
        if self.augment_prompt_for_resize and apply_augmentation:
            image, scale_factor = self._apply_custom_diffusion_augmentation(image)
        
        image_tensor = transforms.ToTensor()(image)
        pixel_values = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image_tensor)
        
        return pixel_values, scale_factor
    
    def _get_prompt_suffix(self, scale_factor: float) -> str:
        """Get prompt suffix based on scale factor."""
        if scale_factor < 0.9:
            return ", very small, far away"
        elif scale_factor > 1.1:
            return ", zoomed in, close up"
        return ""
    
    def __getitem__(self, idx):
        example = {}
        
        # Instance image
        instance_idx = idx % self.num_instance_images
        instance_path = self.instance_images[instance_idx]
        
        pixel_values, scale_factor = self._load_image(instance_path, apply_augmentation=True)
        example["instance_pixel_values"] = pixel_values
        
        # Get prompt
        if self.prompts is not None and instance_idx < len(self.prompts):
            prompt = self.prompts[instance_idx]
        else:
            prompt = self.instance_prompt
        
        # Apply Custom Diffusion prompt augmentation
        if self.augment_prompt_for_resize:
            suffix = self._get_prompt_suffix(scale_factor)
            if suffix:
                prompt = prompt.rstrip() + suffix
        
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
            
            class_pixels, _ = self._load_image(class_path, apply_augmentation=False)
            example["class_pixel_values"] = class_pixels
            
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
        
        return example


def collate_fn_stage1(examples, with_prior_preservation=False):
    """Collate function for Stage 1 (no conditioning)."""
    pixel_values = torch.stack([e["instance_pixel_values"] for e in examples])
    input_ids = torch.stack([e["instance_prompt_ids"] for e in examples])
    
    if with_prior_preservation and "class_pixel_values" in examples[0]:
        class_pixel_values = torch.stack([e["class_pixel_values"] for e in examples])
        class_input_ids = torch.stack([e["class_prompt_ids"] for e in examples])
        
        pixel_values = torch.cat([pixel_values, class_pixel_values], dim=0)
        input_ids = torch.cat([input_ids, class_input_ids], dim=0)
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }


# =============================================================================
# Class Image Generation (for Prior Preservation) - NO ControlNet
# =============================================================================
def generate_class_images_stage1(
    class_images_dir: str,
    class_prompt: str,
    num_class_images: int,
    pretrained_model: str,
    device: str = "cuda",
):
    """
    Generate class images using standard Stable Diffusion (no ControlNet).
    
    For Stage 1 prior preservation, we just need generic class images.
    """
    class_dir = Path(class_images_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    
    existing_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    cur_class_images = len(existing_images)
    
    if cur_class_images >= num_class_images:
        print(f"✅ Found {cur_class_images} class images (needed {num_class_images})")
        return
    
    num_to_generate = num_class_images - cur_class_images
    print(f"📸 Generating {num_to_generate} class images with prompt: '{class_prompt}'")
    
    # Use standard SD pipeline (no ControlNet)
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    for idx in tqdm(range(num_to_generate), desc="Generating class images"):
        image = pipeline(
            prompt=class_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        
        image_path = class_dir / f"class_{cur_class_images + idx:04d}.png"
        image.save(image_path)
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ Generated {num_to_generate} class images")


# =============================================================================
# Training Function - Stage 1: Appearance Pretraining (NO ControlNet)
# =============================================================================
def train(
    data_dir: str = "./data",
    output_dir: str = "./output/stage1-appearance",
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    # DreamBooth-specific parameters
    instance_prompt: str = "a sks humanoid robot",
    class_prompt: str = "a photo of person",
    with_prior_preservation: bool = True,
    prior_loss_weight: float = 1.0,
    num_class_images: int = 100,
    # Training parameters
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    max_train_steps: int = 1000,  # More steps for Stage 1 (no pose guidance)
    checkpointing_steps: int = 200,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = True,
    use_lora: bool = True,
    lora_rank: int = 4,
    custom_diffusion_lora: bool = False,
    train_text_encoder: bool = False,  # Also train text encoder for stronger identity
    # Custom Diffusion-style token training
    train_token_embedding: bool = False,  # Train only the placeholder token embedding
    placeholder_token: str = "sks",  # The rare token to use (ID 48136 in CLIP)
    augment_prompt_for_resize: bool = False,
    seed: int = 42,
    repeats: int = 20,
):
    """
    Stage 1: Appearance Control Pretraining
    
    Train UNet + Text Encoder WITHOUT ControlNet.
    The model learns to associate "sks" with identity from text alone.
    """
    
    # train_token_embedding and train_text_encoder are mutually exclusive
    # If train_token_embedding is enabled, disable train_text_encoder
    if train_token_embedding and train_text_encoder:
        print("⚠️  train_token_embedding enabled - disabling train_text_encoder (they are mutually exclusive)")
        train_text_encoder = False
    
    print("=" * 60)
    print("🎨 Stage 1: Appearance Control Pretraining")
    print("=" * 60)
    print()
    print("🔑 KEY: NO ControlNet - pure text-to-image identity learning!")
    print("   Text Encoder: TRAINED (learning sks token)")
    print("   UNet: TRAINED (learning identity features)")
    print()
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {pretrained_model}")
    print()
    print("DreamBooth Configuration:")
    print(f"  Instance prompt: {instance_prompt}")
    print(f"  Class prompt: {class_prompt}")
    print(f"  Prior preservation: {with_prior_preservation}")
    print(f"  Train text encoder: {train_text_encoder}")
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
    
    # Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16" if mixed_precision else "no",
    )
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    device = accelerator.device
    weight_dtype = torch.float16 if mixed_precision else torch.float32
    
    if seed is not None:
        set_seed(seed)
    
    data_path = Path(data_dir)
    
    if accelerator.is_main_process:
        print_vram("Initial")
    
    # =========================================
    # Generate Class Images (if needed)
    # =========================================
    if with_prior_preservation:
        if accelerator.is_main_process:
            print("=" * 60)
            print("Phase 1: Generating class images for prior preservation")
            print("=" * 60)
            
            class_images_dir = data_path / "class_images"
            
            generate_class_images_stage1(
                class_images_dir=str(class_images_dir),
                class_prompt=class_prompt,
                num_class_images=num_class_images,
                pretrained_model=pretrained_model,
                device="cuda",
            )
        accelerator.wait_for_everyone()
        print()
    
    # =========================================
    # Load Models (NO ControlNet!)
    # =========================================
    print("=" * 60)
    print("Phase 2: Loading models (NO ControlNet)")
    print("=" * 60)
    
    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    print("   ✅ Tokenizer loaded")
    
    # Text encoder (TRAINABLE for Stage 1)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
    
    # Get placeholder token ID (Custom Diffusion style - use existing rare token)
    placeholder_token_id = None
    if train_token_embedding:
        # Encode the placeholder token to get its ID
        placeholder_tokens = tokenizer.encode(placeholder_token, add_special_tokens=False)
        if len(placeholder_tokens) != 1:
            raise ValueError(
                f"Placeholder token '{placeholder_token}' encodes to {len(placeholder_tokens)} tokens. "
                f"Please choose a single-token identifier."
            )
        placeholder_token_id = placeholder_tokens[0]
        print(f"   🎯 Custom Diffusion mode: training token '{placeholder_token}' (ID: {placeholder_token_id})")
    
    if train_text_encoder and use_lora:
        print(f"   🔧 Applying LoRA to text encoder with rank={lora_rank}")
        text_encoder_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "v_proj"],
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        text_encoder.print_trainable_parameters()
        print("   ✅ Text encoder loaded with LoRA (TRAINABLE)")
    elif train_text_encoder:
        text_encoder.to(device)
        print("   ✅ Text encoder loaded (TRAINABLE - full fine-tuning)")
    elif train_token_embedding:
        # Custom Diffusion: freeze all except the token embedding layer
        text_encoder.requires_grad_(False)
        # Unfreeze the token embedding layer (we'll mask gradients later)
        text_encoder.text_model.embeddings.token_embedding.weight.requires_grad = True
        text_encoder.to(device)  # Keep in float32 for training
        print(f"   ✅ Text encoder loaded (frozen except '{placeholder_token}' embedding)")
    else:
        text_encoder.requires_grad_(False)
        text_encoder.to(device, dtype=weight_dtype)
        print("   ✅ Text encoder loaded (frozen)")
    
    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    print("   ✅ VAE loaded (frozen)")
    
    # UNet (TRAINABLE)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    
    if use_lora:
        if custom_diffusion_lora:
            print(f"   🔧 Applying Custom Diffusion LoRA (attn2.to_k, attn2.to_v) with rank={lora_rank}")
            target_modules = ["attn2.to_k", "attn2.to_v"]
        else:
            print(f"   🔧 Applying LoRA with rank={lora_rank}")
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        print("   ✅ UNet loaded with LoRA")
    else:
        unet.to(device)
        print("   ✅ UNet loaded (full fine-tuning)")
    
    # NO ControlNet in Stage 1!
    print("   ⏭️  ControlNet: SKIPPED (Stage 1 - no structural conditioning)")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    print("   ✅ Noise scheduler loaded")
    
    print_vram("After loading models")
    
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_encoder and hasattr(text_encoder, 'gradient_checkpointing_enable'):
            text_encoder.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled")
    
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
    
    dataset = Stage1Dataset(
        instance_images_dir=str(data_path / "instance_images"),
        instance_prompt=instance_prompt,
        tokenizer=tokenizer,
        class_images_dir=str(class_images_dir) if class_images_dir else None,
        class_prompt=class_prompt if with_prior_preservation else None,
        prompts_file=prompts_file,
        resolution=resolution,
        repeats=repeats,
        augment_prompt_for_resize=augment_prompt_for_resize,
    )
    
    def collate_fn_wrapper(examples):
        return collate_fn_stage1(examples, with_prior_preservation=with_prior_preservation)
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_wrapper,
    )
    print(f"   ✅ Dataloader created: {len(dataloader)} batches")
    
    # =========================================
    # Optimizer (for UNet + Text Encoder)
    # =========================================
    print()
    print("=" * 60)
    print("Phase 4: Setting up optimizer")
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
    
    # Get trainable parameters
    if use_lora:
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        if train_text_encoder:
            trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]
        elif train_token_embedding:
            # Add the token embedding layer (we'll mask gradients in training loop)
            trainable_params += [text_encoder.text_model.embeddings.token_embedding.weight]
        lora_lr = learning_rate if learning_rate >= 1e-4 else 1e-4
        print(f"   📊 Training {sum(p.numel() for p in trainable_params):,} LoRA parameters (lr={lora_lr})")
        if train_token_embedding:
            print(f"   📊 + token embedding for '{placeholder_token}' (768 params)")
    else:
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
        if train_text_encoder:
            trainable_params += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
        elif train_token_embedding:
            trainable_params += [text_encoder.text_model.embeddings.token_embedding.weight]
        lora_lr = learning_rate
    
    optimizer = optimizer_class(
        trainable_params,
        lr=lora_lr,
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
    
    # Prepare for distributed training
    if train_text_encoder or train_token_embedding:
        unet, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, dataloader, lr_scheduler
        )
    else:
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )
    
    # Move frozen models to device
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_text_encoder and not train_token_embedding:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    print(f"   ✅ Models prepared for distributed training on {accelerator.num_processes} GPU(s)")
    
    # =========================================
    # Training Loop (NO ControlNet!)
    # =========================================
    print()
    print("=" * 60)
    print("🚀 Phase 5: Starting Training (Stage 1 - No ControlNet)")
    print("=" * 60)
    print(f"  Training: UNet + Text Encoder")
    print(f"  Frozen: VAE")
    print(f"  Disabled: ControlNet (Stage 1)")
    print()
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Stage 1 Training")
    
    unet.train()
    if train_text_encoder:
        text_encoder.train()
    
    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                input_ids = batch["input_ids"]
                
                # Encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict noise with UNet (NO ControlNet residuals!)
                model_pred = unet(
                    noisy_latents.float(),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.float(),
                    # NO down_block_additional_residuals
                    # NO mid_block_additional_residual
                ).sample
                
                # Get target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Calculate loss
                if with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    
                    instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    
                    loss = instance_loss + prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                
                # Custom Diffusion: mask gradients to only update placeholder token
                if train_token_embedding and placeholder_token_id is not None:
                    # Get the token embedding gradients
                    token_embedding = text_encoder.text_model.embeddings.token_embedding
                    if token_embedding.weight.grad is not None:
                        # Create a mask that's 1 only for the placeholder token
                        with torch.no_grad():
                            mask = torch.zeros_like(token_embedding.weight.grad)
                            mask[placeholder_token_id] = 1.0
                            token_embedding.weight.grad.mul_(mask)
                
                if accelerator.sync_gradients:
                    # Collect all trainable parameters for gradient clipping
                    # IMPORTANT: Only call clip_grad_norm_ ONCE per optimizer step
                    params_to_clip = list(unet.parameters())
                    if train_text_encoder:
                        params_to_clip += list(text_encoder.parameters())
                    elif train_token_embedding:
                        params_to_clip.append(text_encoder.text_model.embeddings.token_embedding.weight)
                    
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                effective_loss = loss.item()
                progress_bar.set_postfix(loss=effective_loss)
                
                if global_step % 50 == 0 and accelerator.is_main_process:
                    log_msg = f"\n📊 Step {global_step}/{max_train_steps} | Loss: {effective_loss:.4f}"
                    if with_prior_preservation:
                        log_msg += f" (instance: {instance_loss.item():.4f}, prior: {prior_loss.item():.4f})"
                    print(log_msg)
                    print_vram("Training")
                
                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    unet_to_save = accelerator.unwrap_model(unet)
                    if use_lora:
                        unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet_lora"))
                        print(f"\n💾 UNet LoRA checkpoint saved: {checkpoint_dir}/unet_lora")
                    else:
                        unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet"))
                    
                    if train_text_encoder:
                        text_encoder_to_save = accelerator.unwrap_model(text_encoder)
                        if use_lora:
                            text_encoder_to_save.save_pretrained(os.path.join(checkpoint_dir, "text_encoder_lora"))
                            print(f"💾 Text Encoder LoRA checkpoint saved: {checkpoint_dir}/text_encoder_lora")
                    
                    # Save token embedding if training it (Custom Diffusion style)
                    if train_token_embedding and placeholder_token_id is not None:
                        text_encoder_unwrapped = accelerator.unwrap_model(text_encoder)
                        token_embedding_weight = text_encoder_unwrapped.text_model.embeddings.token_embedding.weight
                        token_embedding_data = {
                            "token_id": placeholder_token_id,
                            "token": placeholder_token,
                            "embedding": token_embedding_weight[placeholder_token_id].detach().cpu(),
                        }
                        torch.save(token_embedding_data, os.path.join(checkpoint_dir, "token_embedding.pt"))
                        print(f"💾 Token embedding checkpoint saved: {checkpoint_dir}/token_embedding.pt")
            
            if global_step >= max_train_steps:
                break
    
    # =========================================
    # Save Final Model
    # =========================================
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print()
        print("=" * 60)
        print("💾 Saving Final Stage 1 Model")
        print("=" * 60)
        
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        if use_lora:
            lora_path = os.path.join(output_dir, "unet_lora")
            unwrapped_unet.save_pretrained(lora_path)
            print(f"✅ UNet LoRA weights saved to: {lora_path}")
            
            if train_text_encoder:
                unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                text_encoder_lora_path = os.path.join(output_dir, "text_encoder_lora")
                unwrapped_text_encoder.save_pretrained(text_encoder_lora_path)
                print(f"✅ Text encoder LoRA weights saved to: {text_encoder_lora_path}")
            
            # Save token embedding if trained (Custom Diffusion style)
            if train_token_embedding and placeholder_token_id is not None:
                text_encoder_unwrapped = accelerator.unwrap_model(text_encoder)
                token_embedding_weight = text_encoder_unwrapped.text_model.embeddings.token_embedding.weight
                token_embedding_data = {
                    "token_id": placeholder_token_id,
                    "token": placeholder_token,
                    "embedding": token_embedding_weight[placeholder_token_id].detach().cpu(),
                }
                token_embedding_path = os.path.join(output_dir, "token_embedding.pt")
                torch.save(token_embedding_data, token_embedding_path)
                print(f"✅ Token embedding saved to: {token_embedding_path}")
            
            print()
            print("=" * 60)
            print("✅ Stage 1 Training Complete!")
            print("=" * 60)
            print()
            print("Next step: Run inference to test:")
            print("```bash")
            print(f"python infer_stage1.py \\")
            print(f"    --prompt \"{instance_prompt}\" \\")
            print(f"    --lora_path {output_dir} \\")
            print(f"    --num_images 4")
            print("```")
        else:
            unet_path = os.path.join(output_dir, "unet")
            unwrapped_unet.save_pretrained(unet_path)
            print(f"✅ UNet saved to: {unet_path}")
    
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Appearance Control Pretraining (No ControlNet)"
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output/stage1-appearance")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    
    # DreamBooth parameters
    parser.add_argument("--instance_prompt", type=str, default="a sks humanoid robot",
                       help="Prompt with rare identifier for the instance")
    parser.add_argument("--class_prompt", type=str, default="a photo of person",
                       help="General class prompt for prior preservation")
    parser.add_argument("--with_prior_preservation", action="store_true",
                       help="Enable prior preservation loss")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0,
                       help="Weight of prior preservation loss")
    parser.add_argument("--num_class_images", type=int, default=100,
                       help="Number of class images for prior preservation")
    
    # Training parameters
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                       help="Max training steps (more for Stage 1 since no pose guidance)")
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20,
                       help="Times to repeat instance images")
    
    # Optimization flags
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_8bit_adam", action="store_true")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA (full fine-tuning, requires more VRAM)")
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="LoRA rank (4-8 typical)")
    parser.add_argument("--custom_diffusion_lora", action="store_true",
                       help="Use Custom Diffusion LoRA targets (attn2.to_k, attn2.to_v only)")
    parser.add_argument("--train_text_encoder", action="store_true",
                       help="Also train text encoder for stronger identity learning")
    # Custom Diffusion-style token training
    parser.add_argument("--train_token_embedding", action="store_true",
                       help="Train only the placeholder token embedding (Custom Diffusion style)")
    parser.add_argument("--placeholder_token", type=str, default="sks",
                       help="The placeholder token to train (default: sks, ID 48136 in CLIP)")
    parser.add_argument("--augment_prompt_for_resize", action="store_true",
                       help="Custom Diffusion: augment prompts for resized images")
    
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
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_8bit_adam=not args.no_8bit_adam,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        custom_diffusion_lora=args.custom_diffusion_lora,
        train_text_encoder=args.train_text_encoder,
        train_token_embedding=args.train_token_embedding,
        placeholder_token=args.placeholder_token,
        augment_prompt_for_resize=args.augment_prompt_for_resize,
        seed=args.seed,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
