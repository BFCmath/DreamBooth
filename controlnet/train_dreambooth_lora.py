#!/usr/bin/env python3
"""
DreamBooth LoRA Training for SD 1.5
Train a LoRA adapter on the UNet to learn identity (e.g., "sks cat").

This LoRA can then be combined with any pretrained ControlNet at inference:
- LoRA: Handles identity ("who is sks cat")
- ControlNet: Handles pose/structure (unchanged)

Usage:
    python train_dreambooth_lora.py --instance_prompt "a photo of sks cat"
    
Inference:
    Load: SD1.5 + LoRA + ControlNet (e.g., OpenPose/Canny)
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
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

# LoRA imports
from peft import LoraConfig, get_peft_model


# =============================================================================
# Dataset
# =============================================================================
class DreamBoothDataset(Dataset):
    """Simple dataset for DreamBooth LoRA training."""
    
    def __init__(
        self,
        instance_images_dir: str,
        instance_prompt: str,
        tokenizer,
        class_images_dir: str = None,
        class_prompt: str = None,
        resolution: int = 512,
        repeats: int = 1,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        
        # Instance images
        self.instance_images = sorted([
            f for f in Path(instance_images_dir).iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        ])
        self.num_instance_images = len(self.instance_images)
        
        if self.num_instance_images == 0:
            raise ValueError(f"No images found in {instance_images_dir}")
        
        print(f"📊 Found {self.num_instance_images} instance images")
        
        # Class images (for prior preservation)
        self.class_images = []
        if class_images_dir and os.path.exists(class_images_dir):
            self.class_images = sorted([
                f for f in Path(class_images_dir).iterdir()
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
            ])
            self.num_class_images = len(self.class_images)
            print(f"📊 Found {self.num_class_images} class images")
        else:
            self.num_class_images = 0
        
        # Dataset length
        self._length = self.num_instance_images * repeats
        if self.num_class_images > 0:
            self._length = max(self._length, self.num_class_images)
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        example = {}
        
        # Instance image
        instance_path = self.instance_images[idx % self.num_instance_images]
        instance_image = Image.open(instance_path).convert("RGB")
        example["instance_images"] = self.transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        
        # Class image (for prior preservation)
        if self.num_class_images > 0:
            class_path = self.class_images[idx % self.num_class_images]
            class_image = Image.open(class_path).convert("RGB")
            example["class_images"] = self.transforms(class_image)
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
    instance_images = [ex["instance_images"] for ex in examples]
    instance_prompt_ids = [ex["instance_prompt_ids"] for ex in examples]
    
    if with_prior_preservation and "class_images" in examples[0]:
        class_images = [ex["class_images"] for ex in examples]
        class_prompt_ids = [ex["class_prompt_ids"] for ex in examples]
        
        pixel_values = torch.stack(instance_images + class_images)
        input_ids = torch.stack(instance_prompt_ids + class_prompt_ids)
    else:
        pixel_values = torch.stack(instance_images)
        input_ids = torch.stack(instance_prompt_ids)
    
    return {"pixel_values": pixel_values, "input_ids": input_ids}


# =============================================================================
# Class Image Generation
# =============================================================================
def generate_class_images(
    class_images_dir: str,
    class_prompt: str,
    num_class_images: int,
    pretrained_model: str,
    device: str = "cuda",
    sample_batch_size: int = 4,
):
    """Generate class images for prior preservation."""
    class_dir = Path(class_images_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    
    existing = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    if len(existing) >= num_class_images:
        print(f"✅ Found {len(existing)} class images (needed {num_class_images})")
        return
    
    num_to_generate = num_class_images - len(existing)
    print(f"📸 Generating {num_to_generate} class images: '{class_prompt}'")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model, torch_dtype=torch.float16, safety_checker=None
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    num_batches = (num_to_generate + sample_batch_size - 1) // sample_batch_size
    generated = 0
    
    for _ in tqdm(range(num_batches), desc="Generating class images"):
        batch_count = min(sample_batch_size, num_to_generate - generated)
        images = pipeline([class_prompt] * batch_count, num_inference_steps=25).images
        for i, img in enumerate(images):
            img.save(class_dir / f"class_{len(existing) + generated + i:04d}.png")
        generated += batch_count
    
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ Generated {num_to_generate} class images")


# =============================================================================
# Training
# =============================================================================
def train(
    data_dir: str = "./data",
    output_dir: str = "./output/dreambooth-lora",
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    instance_prompt: str = "a photo of sks cat",
    class_prompt: str = "a photo of cat",
    with_prior_preservation: bool = True,
    prior_loss_weight: float = 1.0,
    num_class_images: int = 100,
    # LoRA config
    lora_rank: int = 4,
    lora_alpha: int = 4,
    # Training
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,  # Higher LR for LoRA
    max_train_steps: int = 500,
    checkpointing_steps: int = 100,
    mixed_precision: bool = True,
    seed: int = 42,
    repeats: int = 100,
    sample_batch_size: int = 4,
):
    """
    Train DreamBooth LoRA on UNet.
    
    The LoRA learns the identity ("sks cat").
    At inference, combine with any pretrained ControlNet for pose control.
    """
    print("=" * 60)
    print("DreamBooth LoRA Training")
    print("=" * 60)
    print(f"Instance prompt: {instance_prompt}")
    print(f"Class prompt: {class_prompt}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max steps: {max_train_steps}")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if mixed_precision and device == "cuda" else torch.float32
    
    if seed is not None:
        torch.manual_seed(seed)
    
    data_path = Path(data_dir)
    
    # Generate class images if needed
    if with_prior_preservation:
        generate_class_images(
            str(data_path / "class_images"),
            class_prompt,
            num_class_images,
            pretrained_model,
            device,
            sample_batch_size,
        )
    
    # Load models
    print("\n⏳ Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    
    # Add LoRA to UNet
    print("\n⏳ Adding LoRA adapters to UNet...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Attention layers
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.to(device)
    
    print("✅ Models loaded")
    
    # Dataset
    print("\n⏳ Loading dataset...")
    dataset = DreamBoothDataset(
        instance_images_dir=str(data_path / "instance_images"),
        instance_prompt=instance_prompt,
        tokenizer=tokenizer,
        class_images_dir=str(data_path / "class_images") if with_prior_preservation else None,
        class_prompt=class_prompt if with_prior_preservation else None,
        resolution=resolution,
        repeats=repeats,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, with_prior_preservation),
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Optimizer (only LoRA parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("🚀 Starting Training")
    print("=" * 60)
    
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), desc="Training")
    unet.train()
    
    while global_step < max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)
            
            # Encode images
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Predict noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            
            # Loss
            if with_prior_preservation:
                model_pred, model_pred_prior = torch.chunk(model_pred, 2)
                noise, noise_prior = torch.chunk(noise, 2)
                loss = F.mse_loss(model_pred.float(), noise.float())
                prior_loss = F.mse_loss(model_pred_prior.float(), noise_prior.float())
                loss = loss + prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), noise.float())
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps)
            global_step += 1
            
            if global_step % checkpointing_steps == 0:
                unet.save_pretrained(os.path.join(output_dir, f"checkpoint-{global_step}"))
                print(f"\n💾 Saved checkpoint-{global_step}")
            
            if global_step >= max_train_steps:
                break
    
    # Save final LoRA
    print("\n" + "=" * 60)
    print("💾 Saving LoRA weights")
    print("=" * 60)
    unet.save_pretrained(output_dir)
    print(f"✅ LoRA saved to: {output_dir}")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print("\nTo use with ControlNet:")
    print("```python")
    print("from diffusers import StableDiffusionControlNetPipeline, ControlNetModel")
    print("from peft import PeftModel")
    print()
    print("# Load base pipeline with ControlNet")
    print('controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")')
    print('pipe = StableDiffusionControlNetPipeline.from_pretrained(')
    print(f'    "{pretrained_model}", controlnet=controlnet)')
    print()
    print("# Load LoRA weights")
    print(f'pipe.unet = PeftModel.from_pretrained(pipe.unet, "{output_dir}")')
    print()
    print("# Generate with identity + pose control")
    print(f'image = pipe("{instance_prompt}", image=canny_image).images[0]')
    print("```")


def main():
    parser = argparse.ArgumentParser(description="DreamBooth LoRA Training")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output/dreambooth-lora")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks cat")
    parser.add_argument("--class_prompt", type=str, default="a photo of cat")
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--no_mixed_precision", action="store_true")
    
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
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=not args.no_mixed_precision,
        seed=args.seed,
        repeats=args.repeats,
        sample_batch_size=args.sample_batch_size,
    )


if __name__ == "__main__":
    main()
