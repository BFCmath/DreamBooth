#!/usr/bin/env python3
"""
Stage 2: Appearance-Disentangled Pose Control

This script trains the Pose ControlNet while fine-tuning the UNet + Text Encoder
(loaded from Stage 1). The goal is to teach pose control WITHOUT overriding 
the appearance learned in Stage 1.

Key features:
- Loads Stage 1 trained LoRA (UNet + Text Encoder)
- ControlNet is TRAINABLE (pretrained base, learning pose control)
- UNet continues fine-tuning (preserving identity)

This approach:
1. Leverages Stage 1's identity learning
2. Teaches ControlNet to control structure without re-learning appearance
3. Disentangles motion/pose from identity

Based on:
- MagicPose paper: https://arxiv.org/pdf/2311.12052
- ControlNet paper: https://arxiv.org/abs/2302.05543

Dataset Structure Required:
    data/
    ├── instance_images/      # Instance images (3+ images with extractable poses)
    │   ├── 001.png
    │   └── ...
    ├── conditioning/         # Pose images (OpenPose format)
    │   ├── 001.png           # Must match instance image names
    │   └── ...
    └── prompts.txt           # Optional: One prompt per line

Key Hyperparameters:
- stage1_lora_path: Path to Stage 1 trained LoRA weights
- instance_prompt: "a photo of sks person" (same as Stage 1)
- learning_rate: 1e-5 (higher for ControlNet training)
- max_train_steps: 500-800 (fewer steps - leveraging Stage 1)
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
from peft import LoraConfig, get_peft_model, PeftModel

# Local utilities
from utils import print_vram, collate_fn


# =============================================================================
# Dataset for Stage 2 (With Conditioning)
# =============================================================================
class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 pose control training.
    
    Requires conditioning images (poses) for each instance image.
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
        augment_prompt_for_resize: bool = False,
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
        
        # Custom Diffusion augmentation settings
        self.scale_min = 0.4
        self.scale_max = 1.4
        self.pad_fill_color = (128, 128, 128)
    
    def __len__(self):
        return self._length
    
    def _apply_custom_diffusion_augmentation(self, image: Image.Image, cond_image: Image.Image):
        """Apply Custom Diffusion random scale augmentation to BOTH images."""
        import random
        
        scale_factor = random.uniform(self.scale_min, self.scale_max)
        base_size = self.resolution
        
        if scale_factor < 1.0:
            new_size = int(base_size * scale_factor)
            resized_image = image.resize((new_size, new_size), Image.BILINEAR)
            resized_cond = cond_image.resize((new_size, new_size), Image.BILINEAR)
            
            padded_image = Image.new("RGB", (base_size, base_size), self.pad_fill_color)
            padded_cond = Image.new("RGB", (base_size, base_size), self.pad_fill_color)
            
            offset = (base_size - new_size) // 2
            padded_image.paste(resized_image, (offset, offset))
            padded_cond.paste(resized_cond, (offset, offset))
            
            return padded_image, padded_cond, scale_factor
        else:
            new_size = int(base_size * scale_factor)
            resized_image = image.resize((new_size, new_size), Image.BILINEAR)
            resized_cond = cond_image.resize((new_size, new_size), Image.BILINEAR)
            
            offset = (new_size - base_size) // 2
            cropped_image = resized_image.crop((offset, offset, offset + base_size, offset + base_size))
            cropped_cond = resized_cond.crop((offset, offset, offset + base_size, offset + base_size))
            
            return cropped_image, cropped_cond, scale_factor
    
    def _load_image_and_conditioning(self, image_path, cond_dir, apply_augmentation=True):
        """Load an image and its corresponding conditioning image."""
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
        
        # Resize to target resolution
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        cond_image = cond_image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        scale_factor = 1.0
        if self.augment_prompt_for_resize and apply_augmentation:
            image, cond_image, scale_factor = self._apply_custom_diffusion_augmentation(
                image, cond_image
            )
        
        # Convert to tensors
        image_tensor = transforms.ToTensor()(image)
        pixel_values = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image_tensor)
        conditioning_pixel_values = transforms.ToTensor()(cond_image)
        
        return pixel_values, conditioning_pixel_values, scale_factor
    
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
# Class Image Generation (with ControlNet)
# =============================================================================
def generate_class_images_stage2(
    class_images_dir: str,
    class_conditioning_dir: str,
    instance_conditioning_dir: str,
    class_prompt: str,
    num_class_images: int,
    pretrained_model: str,
    controlnet_path: str,
    device: str = "cuda",
):
    """Generate class images using ControlNet for conditioning."""
    class_dir = Path(class_images_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    
    cond_dir = Path(class_conditioning_dir)
    cond_dir.mkdir(parents=True, exist_ok=True)
    
    instance_cond_dir = Path(instance_conditioning_dir)
    
    existing_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    cur_class_images = len(existing_images)
    
    if cur_class_images >= num_class_images:
        existing_cond = list(cond_dir.glob("*.png")) + list(cond_dir.glob("*.jpg"))
        if len(existing_cond) >= num_class_images:
            print(f"✅ Found {cur_class_images} class images + {len(existing_cond)} conditioning")
            return
    
    instance_cond_files = sorted([
        f for f in instance_cond_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    ])
    
    if len(instance_cond_files) == 0:
        raise ValueError(f"No conditioning images found in {instance_conditioning_dir}")
    
    num_to_generate = num_class_images - cur_class_images
    print(f"📸 Generating {num_to_generate} class images with ControlNet conditioning")
    
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    for idx in tqdm(range(num_to_generate), desc="Generating class images"):
        cond_idx = (cur_class_images + idx) % len(instance_cond_files)
        cond_file = instance_cond_files[cond_idx]
        
        cond_image = Image.open(cond_file).convert("RGB")
        
        image = pipeline(
            prompt=class_prompt,
            image=cond_image,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        
        image_path = class_dir / f"class_{cur_class_images + idx:04d}.png"
        image.save(image_path)
        
        cond_path = cond_dir / f"class_{cur_class_images + idx:04d}.png"
        cond_image.save(cond_path)
    
    del pipeline
    del controlnet
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✅ Generated {num_to_generate} class images")


# =============================================================================
# Training Function - Stage 2: Pose Control with Trainable ControlNet
# =============================================================================
def train(
    data_dir: str = "./data",
    output_dir: str = "./output/stage2-pose",
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose",
    stage1_lora_path: str = None,  # Path to Stage 1 trained LoRA
    # DreamBooth-specific parameters
    instance_prompt: str = "a photo of sks person",
    class_prompt: str = "a photo of person",
    with_prior_preservation: bool = True,
    prior_loss_weight: float = 1.0,
    num_class_images: int = 100,
    sample_batch_size: int = 4,
    # Training parameters
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,  # Higher LR for ControlNet training
    controlnet_learning_rate: float = 1e-5,  # Separate LR for ControlNet
    max_train_steps: int = 600,  # Fewer steps - leveraging Stage 1
    checkpointing_steps: int = 200,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = True,
    use_lora: bool = True,
    lora_rank: int = 4,
    custom_diffusion_lora: bool = False,
    train_text_encoder: bool = True,  # Continue training text encoder
    train_controlnet: bool = True,  # Train ControlNet (Stage 2 key feature!)
    augment_prompt_for_resize: bool = False,
    seed: int = 42,
    repeats: int = 20,
):
    """
    Stage 2: Appearance-Disentangled Pose Control
    
    Train ControlNet for pose control while preserving identity from Stage 1.
    Loads Stage 1 LoRA weights for UNet + Text Encoder.
    """
    
    print("=" * 60)
    print("🎭 Stage 2: Appearance-Disentangled Pose Control")
    print("=" * 60)
    print()
    print("🔑 KEY: ControlNet is TRAINABLE (learning pose control)!")
    print("   Loading Stage 1 LoRA for identity preservation")
    print()
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {pretrained_model}")
    print(f"ControlNet: {controlnet_model}")
    print(f"Stage 1 LoRA: {stage1_lora_path}")
    print()
    print("Stage 2 Configuration:")
    print(f"  Instance prompt: {instance_prompt}")
    print(f"  Class prompt: {class_prompt}")
    print(f"  Prior preservation: {with_prior_preservation}")
    print(f"  Train ControlNet: {train_controlnet}")
    print(f"  Train text encoder: {train_text_encoder}")
    print()
    print("Training Configuration:")
    print(f"  Resolution: {resolution}")
    print(f"  Batch size: {train_batch_size}")
    print(f"  UNet/Text LR: {learning_rate}")
    print(f"  ControlNet LR: {controlnet_learning_rate}")
    print(f"  Max steps: {max_train_steps}")
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
            class_conditioning_dir = data_path / "class_conditioning"
            instance_conditioning_dir = data_path / "conditioning"
            
            generate_class_images_stage2(
                class_images_dir=str(class_images_dir),
                class_conditioning_dir=str(class_conditioning_dir),
                instance_conditioning_dir=str(instance_conditioning_dir),
                class_prompt=class_prompt,
                num_class_images=num_class_images,
                pretrained_model=pretrained_model,
                controlnet_path=controlnet_model,
                device="cuda",
            )
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
    
    # Text encoder - load Stage 1 LoRA if available
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
    
    if stage1_lora_path and os.path.exists(os.path.join(stage1_lora_path, "text_encoder_lora")):
        text_encoder_lora_path = os.path.join(stage1_lora_path, "text_encoder_lora")
        print(f"   🔄 Loading Stage 1 Text Encoder LoRA from {text_encoder_lora_path}")
        text_encoder = PeftModel.from_pretrained(text_encoder, text_encoder_lora_path)
        
        if train_text_encoder:
            # Unfreeze for continued training
            for param in text_encoder.parameters():
                param.requires_grad = True
            print("   ✅ Text encoder loaded with Stage 1 LoRA (TRAINABLE)")
        else:
            text_encoder.requires_grad_(False)
            print("   ✅ Text encoder loaded with Stage 1 LoRA (frozen)")
    elif train_text_encoder and use_lora:
        print(f"   🔧 Applying fresh LoRA to text encoder with rank={lora_rank}")
        text_encoder_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "v_proj"],
        )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        print("   ✅ Text encoder loaded with fresh LoRA (TRAINABLE)")
    else:
        text_encoder.requires_grad_(False)
        text_encoder.to(device, dtype=weight_dtype)
        print("   ✅ Text encoder loaded (frozen)")
    
    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    print("   ✅ VAE loaded (frozen)")
    
    # UNet - load Stage 1 LoRA if available
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    
    if stage1_lora_path and os.path.exists(os.path.join(stage1_lora_path, "unet_lora")):
        unet_lora_path = os.path.join(stage1_lora_path, "unet_lora")
        print(f"   🔄 Loading Stage 1 UNet LoRA from {unet_lora_path}")
        unet = PeftModel.from_pretrained(unet, unet_lora_path)
        
        # Unfreeze for continued training
        for param in unet.parameters():
            param.requires_grad = True
        unet.print_trainable_parameters()
        print("   ✅ UNet loaded with Stage 1 LoRA (TRAINABLE)")
    elif use_lora:
        if custom_diffusion_lora:
            print(f"   🔧 Applying Custom Diffusion LoRA (attn2.to_k, attn2.to_v) with rank={lora_rank}")
            target_modules = ["attn2.to_k", "attn2.to_v"]
        else:
            print(f"   🔧 Applying fresh LoRA with rank={lora_rank}")
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        print("   ✅ UNet loaded with fresh LoRA")
    else:
        unet.to(device)
        print("   ✅ UNet loaded (full fine-tuning)")
    
    # ControlNet (TRAINABLE in Stage 2!)
    controlnet = ControlNetModel.from_pretrained(controlnet_model)
    
    if train_controlnet:
        # Keep ControlNet trainable  
        controlnet.requires_grad_(True)
        print(f"   ✅ ControlNet loaded from {controlnet_model} (TRAINABLE)")
        
        # Count trainable params
        controlnet_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        print(f"   📊 ControlNet trainable parameters: {controlnet_params:,}")
    else:
        controlnet.requires_grad_(False)
        controlnet.to(device, dtype=weight_dtype)
        print(f"   ✅ ControlNet loaded from {controlnet_model} (frozen)")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    print("   ✅ Noise scheduler loaded")
    
    print_vram("After loading models")
    
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_controlnet:
            controlnet.enable_gradient_checkpointing()
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
    class_conditioning_dir = str(data_path / "class_conditioning") if with_prior_preservation else None
    
    dataset = Stage2Dataset(
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
    # Optimizer (for UNet, Text Encoder, and ControlNet)
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
    
    # Collect trainable parameters with different learning rates
    param_groups = []
    
    # UNet parameters
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    if unet_params:
        unet_lr = learning_rate if learning_rate >= 1e-4 else 1e-4
        param_groups.append({"params": unet_params, "lr": unet_lr})
        print(f"   📊 UNet: {sum(p.numel() for p in unet_params):,} params (lr={unet_lr})")
    
    # Text encoder parameters
    if train_text_encoder:
        text_params = [p for p in text_encoder.parameters() if p.requires_grad]
        if text_params:
            param_groups.append({"params": text_params, "lr": learning_rate})
            print(f"   📊 Text Encoder: {sum(p.numel() for p in text_params):,} params (lr={learning_rate})")
    
    # ControlNet parameters (with separate learning rate)
    if train_controlnet:
        controlnet_params = [p for p in controlnet.parameters() if p.requires_grad]
        if controlnet_params:
            param_groups.append({"params": controlnet_params, "lr": controlnet_learning_rate})
            print(f"   📊 ControlNet: {sum(p.numel() for p in controlnet_params):,} params (lr={controlnet_learning_rate})")
    
    optimizer = optimizer_class(
        param_groups,
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
    models_to_prepare = [unet, optimizer, dataloader, lr_scheduler]
    if train_text_encoder:
        models_to_prepare.insert(1, text_encoder)
    if train_controlnet:
        models_to_prepare.insert(1, controlnet)
    
    prepared = accelerator.prepare(*models_to_prepare)
    
    # Unpack prepared models
    idx = 0
    unet = prepared[idx]; idx += 1
    if train_controlnet:
        controlnet = prepared[idx]; idx += 1
    if train_text_encoder:
        text_encoder = prepared[idx]; idx += 1
    optimizer = prepared[idx]; idx += 1
    dataloader = prepared[idx]; idx += 1
    lr_scheduler = prepared[idx]
    
    # Move frozen models to device
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if not train_controlnet:
        controlnet.to(accelerator.device, dtype=weight_dtype)
    
    print(f"   ✅ Models prepared for distributed training on {accelerator.num_processes} GPU(s)")
    
    # =========================================
    # Training Loop
    # =========================================
    print()
    print("=" * 60)
    print("🚀 Phase 5: Starting Training (Stage 2 - Pose Control)")
    print("=" * 60)
    print(f"  Training: UNet + ControlNet" + (" + Text Encoder" if train_text_encoder else ""))
    print(f"  Frozen: VAE")
    print()
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Stage 2 Training")
    
    unet.train()
    if train_controlnet:
        controlnet.train()
    if train_text_encoder:
        text_encoder.train()
    
    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
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
                
                # Get ControlNet output (TRAINABLE in Stage 2!)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(dtype=weight_dtype),
                    controlnet_cond=conditioning_pixel_values,
                    return_dict=False,
                )
                
                # Predict noise with UNet using ControlNet guidance
                model_pred = unet(
                    noisy_latents.float(),
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
                
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters())
                    if train_text_encoder:
                        params_to_clip += list(text_encoder.parameters())
                    if train_controlnet:
                        params_to_clip += list(controlnet.parameters())
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
                    
                    # Save UNet
                    unet_to_save = accelerator.unwrap_model(unet)
                    if use_lora or stage1_lora_path:
                        unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet_lora"))
                        print(f"\n💾 UNet LoRA checkpoint saved")
                    else:
                        unet_to_save.save_pretrained(os.path.join(checkpoint_dir, "unet"))
                    
                    # Save ControlNet
                    if train_controlnet:
                        controlnet_to_save = accelerator.unwrap_model(controlnet)
                        controlnet_to_save.save_pretrained(os.path.join(checkpoint_dir, "controlnet"))
                        print(f"💾 ControlNet checkpoint saved")
                    
                    # Save Text Encoder
                    if train_text_encoder:
                        text_encoder_to_save = accelerator.unwrap_model(text_encoder)
                        text_encoder_to_save.save_pretrained(os.path.join(checkpoint_dir, "text_encoder_lora"))
                        print(f"💾 Text Encoder LoRA checkpoint saved")
            
            if global_step >= max_train_steps:
                break
    
    # =========================================
    # Save Final Model
    # =========================================
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print()
        print("=" * 60)
        print("💾 Saving Final Stage 2 Model")
        print("=" * 60)
        
        # Save UNet
        unwrapped_unet = accelerator.unwrap_model(unet)
        if use_lora or stage1_lora_path:
            lora_path = os.path.join(output_dir, "unet_lora")
            unwrapped_unet.save_pretrained(lora_path)
            print(f"✅ UNet LoRA weights saved to: {lora_path}")
        else:
            unet_path = os.path.join(output_dir, "unet")
            unwrapped_unet.save_pretrained(unet_path)
            print(f"✅ UNet saved to: {unet_path}")
        
        # Save ControlNet
        if train_controlnet:
            unwrapped_controlnet = accelerator.unwrap_model(controlnet)
            controlnet_path = os.path.join(output_dir, "controlnet")
            unwrapped_controlnet.save_pretrained(controlnet_path)
            print(f"✅ Trained ControlNet saved to: {controlnet_path}")
        
        # Save Text Encoder
        if train_text_encoder:
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder_path = os.path.join(output_dir, "text_encoder_lora")
            unwrapped_text_encoder.save_pretrained(text_encoder_path)
            print(f"✅ Text encoder LoRA saved to: {text_encoder_path}")
        
        print()
        print("=" * 60)
        print("✅ Stage 2 Training Complete!")
        print("=" * 60)
        print()
        print("To use your trained model for inference:")
        print("```python")
        print("from diffusers import StableDiffusionControlNetPipeline, ControlNetModel")
        print("from peft import PeftModel")
        print()
        print(f'controlnet = ControlNetModel.from_pretrained("{os.path.join(output_dir, "controlnet")}")')
        print(f'pipe = StableDiffusionControlNetPipeline.from_pretrained("{pretrained_model}", controlnet=controlnet)')
        if use_lora or stage1_lora_path:
            print(f'pipe.unet = PeftModel.from_pretrained(pipe.unet, "{os.path.join(output_dir, "unet_lora")}")')
        if train_text_encoder:
            print(f'pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, "{os.path.join(output_dir, "text_encoder_lora")}")')
        print(f'image = pipe("{instance_prompt}", image=pose_image).images[0]')
        print("```")
    
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Appearance-Disentangled Pose Control (Train ControlNet)"
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./output/stage2-pose")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/control_v11p_sd15_openpose",
                       help="Pretrained ControlNet to start from")
    parser.add_argument("--stage1_lora_path", type=str, default=None,
                       help="Path to Stage 1 trained LoRA weights (contains unet_lora/ and text_encoder_lora/)")
    
    # DreamBooth parameters
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person",
                       help="Prompt with rare identifier (same as Stage 1)")
    parser.add_argument("--class_prompt", type=str, default="a photo of person",
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
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for UNet/Text Encoder")
    parser.add_argument("--controlnet_learning_rate", type=float, default=1e-5,
                       help="Learning rate for ControlNet")
    parser.add_argument("--max_train_steps", type=int, default=600,
                       help="Max training steps (fewer for Stage 2)")
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20,
                       help="Times to repeat instance images")
    
    # Optimization flags
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_8bit_adam", action="store_true")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA for UNet (full fine-tuning)")
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="LoRA rank")
    parser.add_argument("--custom_diffusion_lora", action="store_true",
                       help="Use Custom Diffusion LoRA targets")
    parser.add_argument("--no_train_text_encoder", action="store_true",
                       help="Disable text encoder training")
    parser.add_argument("--no_train_controlnet", action="store_true",
                       help="Freeze ControlNet (NOT recommended for Stage 2)")
    parser.add_argument("--augment_prompt_for_resize", action="store_true",
                       help="Custom Diffusion: augment prompts for resized images")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        controlnet_model=args.controlnet_model,
        stage1_lora_path=args.stage1_lora_path,
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
        controlnet_learning_rate=args.controlnet_learning_rate,
        max_train_steps=args.max_train_steps,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_8bit_adam=not args.no_8bit_adam,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        custom_diffusion_lora=args.custom_diffusion_lora,
        train_text_encoder=not args.no_train_text_encoder,
        train_controlnet=not args.no_train_controlnet,
        augment_prompt_for_resize=args.augment_prompt_for_resize,
        seed=args.seed,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
