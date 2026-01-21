#!/usr/bin/env python3
"""
Custom Diffusion: Multi-Concept Customization of Text-to-Image Diffusion

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
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model

# Local utilities
from utils import print_vram, extract_conditioning, collate_fn


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
        augment_prompt_for_resize: bool = False,  # Custom Diffusion: augment prompts for small/cropped images
    ):
        self.instance_images_dir = Path(instance_images_dir)
        self.conditioning_dir = Path(conditioning_dir)
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
        
        # Custom Diffusion random scale range (0.4 to 1.4)
        self.scale_min = 0.4
        self.scale_max = 1.4
        
        # Fill color for padding (gray = 128/255 ≈ 0.5)
        self.pad_fill_color = (128, 128, 128)
    
    def __len__(self):
        return self._length
    
    def _apply_custom_diffusion_augmentation(self, image: Image.Image, cond_image: Image.Image):
        """
        Apply Custom Diffusion random scale augmentation to BOTH image and conditioning.
        
        From the paper:
        - Random scale between 0.4 and 1.4
        - If scale < 1.0: Shrink image, pad with grey → "very small, far away"
        - If scale > 1.0: Enlarge and center crop → "zoomed in, close up"
        
        CRITICAL: Same transform is applied to BOTH image and conditioning!
        
        Returns:
            augmented_image: PIL Image at resolution x resolution
            augmented_cond: PIL Image at resolution x resolution  
            scale_factor: The random scale used (for prompt augmentation)
        """
        import random
        
        # Sample random scale factor
        scale_factor = random.uniform(self.scale_min, self.scale_max)
        
        # Both images should be resized to same base size first
        # We'll work at the target resolution
        base_size = self.resolution
        
        if scale_factor < 1.0:
            # SHRINK: Make object appear smaller/farther
            # Shrink the image, then pad to fill resolution
            new_size = int(base_size * scale_factor)
            
            # Resize both images to smaller size
            resized_image = image.resize((new_size, new_size), Image.BILINEAR)
            resized_cond = cond_image.resize((new_size, new_size), Image.BILINEAR)
            
            # Create padded images with grey background
            padded_image = Image.new("RGB", (base_size, base_size), self.pad_fill_color)
            padded_cond = Image.new("RGB", (base_size, base_size), self.pad_fill_color)
            
            # Center the resized image
            offset = (base_size - new_size) // 2
            padded_image.paste(resized_image, (offset, offset))
            padded_cond.paste(resized_cond, (offset, offset))
            
            return padded_image, padded_cond, scale_factor
            
        else:
            # ENLARGE & CROP: Make object appear larger/closer (zoomed in)
            # Enlarge the image, then center crop to resolution
            new_size = int(base_size * scale_factor)
            
            # Resize both images to larger size
            resized_image = image.resize((new_size, new_size), Image.BILINEAR)
            resized_cond = cond_image.resize((new_size, new_size), Image.BILINEAR)
            
            # Center crop back to resolution
            offset = (new_size - base_size) // 2
            cropped_image = resized_image.crop((offset, offset, offset + base_size, offset + base_size))
            cropped_cond = resized_cond.crop((offset, offset, offset + base_size, offset + base_size))
            
            return cropped_image, cropped_cond, scale_factor
    
    def _load_image_and_conditioning(self, image_path, cond_dir, apply_augmentation=True):
        """
        Load an image and its corresponding conditioning image.
        
        If augment_prompt_for_resize is enabled, applies Custom Diffusion 
        random scale augmentation to BOTH images with the same transform.
        
        Returns:
            pixel_values: Transformed image tensor
            conditioning_pixel_values: Transformed conditioning tensor
            scale_factor: The scale factor used (for prompt augmentation)
        """
        image = Image.open(image_path).convert("RGB")
        
        # Find conditioning image
        cond_path = cond_dir / image_path.name
        if not cond_path.exists():
            cond_files = list(cond_dir.glob(f"{image_path.stem}.*"))
            if cond_files:
                cond_path = cond_files[0]
            else:
                raise FileNotFoundError(f"No conditioning image for {image_path.name} in {cond_dir}")
        
        cond_image = Image.open(cond_path).convert("RGB")
        
        # First, resize both images to the target resolution (base preprocessing)
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        cond_image = cond_image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        # Apply Custom Diffusion augmentation if enabled
        scale_factor = 1.0  # No augmentation by default
        if self.augment_prompt_for_resize and apply_augmentation:
            image, cond_image, scale_factor = self._apply_custom_diffusion_augmentation(
                image, cond_image
            )
        
        # Convert to tensors
        # Image: normalize to [-1, 1] for diffusion model
        image_tensor = transforms.ToTensor()(image)
        pixel_values = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image_tensor)
        
        # Conditioning: just [0, 1] range
        conditioning_pixel_values = transforms.ToTensor()(cond_image)
        
        return pixel_values, conditioning_pixel_values, scale_factor
    
    def _get_prompt_suffix(self, scale_factor: float) -> str:
        """
        Get prompt suffix based on scale factor (Custom Diffusion technique).
        
        - scale < 1.0: Object is shrunk → ", very small, far away"
        - scale > 1.0: Object is zoomed → ", zoomed in, close up"
        - scale ≈ 1.0: No suffix
        """
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
        
        pixel_values, conditioning_pixel_values, scale_factor = self._load_image_and_conditioning(
            instance_path, self.conditioning_dir, apply_augmentation=True
        )
        
        example["instance_pixel_values"] = pixel_values
        example["instance_conditioning"] = conditioning_pixel_values
        
        # Get prompt
        if self.prompts is not None and instance_idx < len(self.prompts):
            prompt = self.prompts[instance_idx]
        else:
            prompt = self.instance_prompt
        
        # Apply Custom Diffusion prompt augmentation based on scale factor
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
        # Note: Class images don't get augmentation to maintain stable regularization
        if self.num_class_images > 0:
            class_idx = idx % self.num_class_images
            class_path = self.class_images[class_idx]
            
            if self.class_conditioning_dir is None:
                raise ValueError(
                    "Prior preservation requires class_conditioning_dir! "
                    "Run training with --with_prior_preservation to auto-generate class images + conditioning."
                )
            
            class_pixels, class_cond, _ = self._load_image_and_conditioning(
                class_path, self.class_conditioning_dir, apply_augmentation=False
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


# =============================================================================
# Class Image Generation (for Prior Preservation)
# =============================================================================
def generate_class_images(
    class_images_dir: str,
    class_conditioning_dir: str,
    instance_conditioning_dir: str,  # NEW: Use instance conditioning for class generation
    class_prompt: str,
    num_class_images: int,
    pretrained_model: str,
    controlnet_path: str,  # REQUIRED: Need ControlNet for conditioned generation
    device: str = "cuda",
    sample_batch_size: int = 1,  # Lower default for ControlNet pipeline
):
    """
    Generate class images using the SAME conditioning as instance images.
    
    This is crucial for proper DreamBooth + ControlNet training:
    - Instance: (instance_image, pose_i, "sks cat") 
    - Class:   (class_image, pose_i, "cat")  <- SAME pose!
    
    The prior preservation loss then teaches:
    - "sks" modifier means this specific identity
    - Pose control is preserved because both use the same conditioning
    """
    import shutil
    
    class_dir = Path(class_images_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    
    cond_dir = Path(class_conditioning_dir)
    cond_dir.mkdir(parents=True, exist_ok=True)
    
    instance_cond_dir = Path(instance_conditioning_dir)
    
    # Count existing images
    existing_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    cur_class_images = len(existing_images)
    
    if cur_class_images >= num_class_images:
        existing_cond = list(cond_dir.glob("*.png")) + list(cond_dir.glob("*.jpg"))
        if len(existing_cond) >= num_class_images:
            print(f"✅ Found {cur_class_images} class images + {len(existing_cond)} conditioning (needed {num_class_images})")
            return
    
    # Load instance conditioning images
    instance_cond_files = sorted([
        f for f in instance_cond_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    ])
    
    if len(instance_cond_files) == 0:
        raise ValueError(f"No conditioning images found in {instance_conditioning_dir}")
    
    num_to_generate = num_class_images - cur_class_images
    print(f"📸 Generating {num_to_generate} class images with prompt: '{class_prompt}'")
    print(f"   🎯 Using instance conditioning for proper prior preservation!")
    print(f"   📂 Instance conditioning dir: {instance_conditioning_dir}")
    print(f"   📂 ControlNet: {controlnet_path}")
    
    # Load ControlNet pipeline for conditioned generation
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    generated = 0
    
    for idx in tqdm(range(num_to_generate), desc="Generating class images with instance conditioning"):
        # Cycle through instance conditioning images
        cond_idx = (cur_class_images + idx) % len(instance_cond_files)
        cond_file = instance_cond_files[cond_idx]
        
        # Load conditioning image
        cond_image = Image.open(cond_file).convert("RGB")
        
        # Generate class image using the SAME conditioning as instance
        image = pipeline(
            prompt=class_prompt,
            image=cond_image,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        
        # Save class image
        class_image_idx = cur_class_images + idx
        image_path = class_dir / f"class_{class_image_idx:04d}.png"
        image.save(image_path)
        
        # Copy/save the conditioning (same as instance conditioning)
        cond_path = cond_dir / f"class_{class_image_idx:04d}.png"
        cond_image.save(cond_path)  # Save the conditioning we used
        
        generated += 1
    
    del pipeline
    del controlnet
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ Generated {num_to_generate} class images using instance conditioning")


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
    use_lora: bool = True,  # Use LoRA for memory-efficient training
    lora_rank: int = 4,  # LoRA rank (4-8 typical for DreamBooth)
    custom_diffusion_lora: bool = False,  # Custom Diffusion: only target attn2.to_k, attn2.to_v
    train_text_encoder: bool = False,  # Also train text encoder for stronger identity
    # Custom Diffusion-style token training
    train_token_embedding: bool = False,  # Train only the placeholder token embedding
    placeholder_token: str = "sks",  # The rare token to use (ID 48136 in CLIP)
    augment_prompt_for_resize: bool = False,  # Custom Diffusion: augment prompts for resized images
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
    
    # Setup Accelerator for multi-GPU training
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16" if mixed_precision else "no",
    )
    
    # Only create output dir on main process
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
        # Only generate class images on main process
        if accelerator.is_main_process:
            print("=" * 60)
            print("Phase 1: Generating class images for prior preservation")
            print("=" * 60)
            
            class_images_dir = data_path / "class_images"
            class_conditioning_dir = data_path / "class_conditioning"
            instance_conditioning_dir = data_path / "conditioning"
            
            generate_class_images(
                class_images_dir=str(class_images_dir),
                class_conditioning_dir=str(class_conditioning_dir),
                instance_conditioning_dir=str(instance_conditioning_dir),
                class_prompt=class_prompt,
                num_class_images=num_class_images,
                pretrained_model=pretrained_model,
                controlnet_path=controlnet_model,
                device="cuda",  # Use cuda directly for class generation
                sample_batch_size=sample_batch_size,
            )
        # Wait for class images to be generated before all processes continue
        accelerator.wait_for_everyone()
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
    
    # Text encoder
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
    
    # Apply LoRA to text encoder if train_text_encoder is enabled
    if train_text_encoder and use_lora:
        print(f"   🔧 Applying LoRA to text encoder with rank={lora_rank}")
        text_encoder_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "v_proj"],  # Attention layers in CLIP
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        text_encoder.print_trainable_parameters()
        print("   ✅ Text encoder loaded with LoRA (TRAINABLE)")
    elif train_text_encoder:
        # Full fine-tuning of text encoder
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
    
    # UNet (TRAINABLE - this is where identity learning happens!)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    
    # Apply LoRA for memory-efficient training
    if use_lora:
        # Custom Diffusion paper: only train cross-attention K and V projections
        # This is more parameter-efficient and works well for concept learning
        if custom_diffusion_lora:
            print(f"   🔧 Applying Custom Diffusion LoRA (attn2.to_k, attn2.to_v) with rank={lora_rank}")
            target_modules = ["attn2.to_k", "attn2.to_v"]  # Cross-attention only
        else:
            print(f"   🔧 Applying LoRA with rank={lora_rank}")
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]  # All attention layers
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,  # Typically same as rank
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        if custom_diffusion_lora:
            print("   ✅ UNet loaded with Custom Diffusion LoRA (cross-attention K,V only)")
        else:
            print("   ✅ UNet loaded with LoRA (memory-efficient training)")
    else:
        unet.to(device)  # Keep in float32 for training stability
        print("   ✅ UNet loaded (full fine-tuning)")
    
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
        augment_prompt_for_resize=augment_prompt_for_resize,
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
    
    # Get trainable parameters (LoRA only trains a subset)
    if use_lora:
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        if train_text_encoder:
            trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]
        elif train_token_embedding:
            # Add the token embedding layer (we'll mask gradients in training loop)
            trainable_params += [text_encoder.text_model.embeddings.token_embedding.weight]
        # LoRA typically uses higher learning rate
        lora_lr = learning_rate if learning_rate >= 1e-4 else 1e-4
        print(f"   📊 Training {sum(p.numel() for p in trainable_params):,} LoRA parameters (lr={lora_lr})")
        if train_token_embedding:
            print(f"   📊 + token embedding for '{placeholder_token}' (768 params)")
    else:
        trainable_params = list(unet.parameters())
        if train_text_encoder:
            trainable_params += list(text_encoder.parameters())
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
    controlnet.to(accelerator.device, dtype=weight_dtype)
    
    print(f"   ✅ Models prepared for distributed training on {accelerator.num_processes} GPU(s)")
    
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
            with accelerator.accumulate(unet):
                # Data is already on device from accelerator
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                conditioning_pixel_values = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
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
                
                # Get ControlNet output (frozen, no gradients)
                # Cast to weight_dtype since ControlNet is in fp16
                with torch.no_grad():
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states.to(dtype=weight_dtype),
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
                
                # Backward with accelerator
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
                
                # Update weights (accelerator handles gradient sync)
                if accelerator.sync_gradients:
                    # Collect all trainable parameters for gradient clipping
                    # IMPORTANT: Only call clip_grad_norm_ ONCE per optimizer step
                    # (multiple calls cause "unscale_() already called" error with mixed precision)
                    params_to_clip = list(unet.parameters())
                    if train_token_embedding:
                        params_to_clip.append(text_encoder.text_model.embeddings.token_embedding.weight)
                    
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            
            # Logging (only update progress on sync)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                effective_loss = loss.item()
                progress_bar.set_postfix(loss=effective_loss)
                
                # Periodic logging
                if global_step % 50 == 0 and accelerator.is_main_process:
                    log_msg = f"\n📊 Step {global_step}/{max_train_steps} | Loss: {effective_loss:.4f}"
                    if with_prior_preservation:
                        log_msg += f" (instance: {instance_loss.item():.4f}, prior: {prior_loss.item():.4f})"
                    print(log_msg)
                    print_vram("Training")
                
                # Save checkpoint (only on main process)
                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    # Unwrap and save the UNet/LoRA
                    unet_to_save = accelerator.unwrap_model(unet)
                    if use_lora:
                        unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet_lora"))
                        print(f"\n💾 LoRA checkpoint saved: {checkpoint_dir}/unet_lora")
                    else:
                        unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet"))
                        print(f"\n💾 UNet checkpoint saved: {checkpoint_dir}")
                    
                    # Save token embedding if training it
                    if train_token_embedding and placeholder_token_id is not None:
                        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                        token_embedding_path = os.path.join(checkpoint_dir, "token_embedding.pt")
                        torch.save({
                            "token": placeholder_token,
                            "token_id": placeholder_token_id,
                            "embedding": unwrapped_text_encoder.text_model.embeddings.token_embedding.weight[placeholder_token_id].cpu(),
                        }, token_embedding_path)
                        print(f"💾 Token embedding checkpoint saved: {token_embedding_path}")
            
            if global_step >= max_train_steps:
                break
    
    # =========================================
    # Save Final Model
    # =========================================
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print()
        print("=" * 60)
        print("💾 Saving Final Model")
        print("=" * 60)
        
        # Unwrap the trained UNet
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        if use_lora:
            # Save LoRA weights (small ~5MB files)
            lora_path = os.path.join(output_dir, "unet_lora")
            unwrapped_unet.save_pretrained(lora_path)
            print(f"✅ UNet LoRA weights saved to: {lora_path}")
            
            # Save text encoder LoRA if trained
            if train_text_encoder:
                unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                text_encoder_lora_path = os.path.join(output_dir, "text_encoder_lora")
                unwrapped_text_encoder.save_pretrained(text_encoder_lora_path)
                print(f"✅ Text encoder LoRA weights saved to: {text_encoder_lora_path}")
            
            # Save token embedding if trained (Custom Diffusion style)
            token_embedding_path = None
            if train_token_embedding and placeholder_token_id is not None:
                unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                token_embedding_path = os.path.join(output_dir, "token_embedding.pt")
                torch.save({
                    "token": placeholder_token,
                    "token_id": placeholder_token_id,
                    "embedding": unwrapped_text_encoder.text_model.embeddings.token_embedding.weight[placeholder_token_id].cpu(),
                }, token_embedding_path)
                print(f"✅ Token embedding saved to: {token_embedding_path}")
            
            print()
            print("=" * 60)
            print("✅ Training Complete!")
            print("=" * 60)
            print()
            print("To use your trained LoRA model:")
            print("```python")
            print("from diffusers import StableDiffusionControlNetPipeline, ControlNetModel")
            print("from peft import PeftModel")
            print("import torch")
            print(f'pipe = StableDiffusionControlNetPipeline.from_pretrained("{pretrained_model}")')
            print(f'pipe.unet = PeftModel.from_pretrained(pipe.unet, "{lora_path}")')
            if train_text_encoder:
                print(f'pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, "{text_encoder_lora_path}")')
            if train_token_embedding:
                print(f'# Load trained token embedding')
                print(f'token_data = torch.load("{token_embedding_path}")')
                print(f'pipe.text_encoder.text_model.embeddings.token_embedding.weight.data[token_data["token_id"]] = token_data["embedding"].to(pipe.device)')
            print(f'pipe.controlnet = ControlNetModel.from_pretrained("{controlnet_model}")')
            print(f'image = pipe("{instance_prompt}", image=conditioning_image).images[0]')
            print("```")
        else:
            # Save the full pipeline with trained UNet
            from diffusers import StableDiffusionControlNetPipeline
            
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model,
                unet=unwrapped_unet,
                controlnet=controlnet,
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                safety_checker=None,
            )
            pipeline.save_pretrained(output_dir)
            print(f"✅ Full pipeline saved to: {output_dir}")
            
            # Also save just the UNet separately for easy loading
            unwrapped_unet.save_pretrained(os.path.join(output_dir, "unet_trained"))
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
    
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(
        description="DreamBooth + ControlNet (UNet Training Mode) - trains UNet for identity learning"
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output/dreambooth-unet-controlnet")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11p_sd15_openpose",
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
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA (full UNet fine-tuning, requires more VRAM)")
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="LoRA rank (4-8 typical, higher = more params)")
    parser.add_argument("--custom_diffusion_lora", action="store_true",
                       help="Use Custom Diffusion LoRA targets (attn2.to_k, attn2.to_v only)")
    parser.add_argument("--train_text_encoder", action="store_true",
                       help="Also train text encoder for stronger identity learning")
    parser.add_argument("--augment_prompt_for_resize", action="store_true",
                       help="Custom Diffusion: append ', very small' or ', zoomed in' to prompts for resized images")
    
    # Custom Diffusion style token training
    parser.add_argument("--train_token_embedding", action="store_true",
                       help="Train only the placeholder token embedding (Custom Diffusion style)")
    parser.add_argument("--placeholder_token", type=str, default="sks",
                       help="Placeholder token to train (must be a single token, e.g. 'sks' = ID 48136)")
    
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
