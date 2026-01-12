#!/usr/bin/env python3
"""
DreamBooth Training Script - Based on HuggingFace Diffusers Official Implementation
Optimized for 16GB VRAM (Kaggle P100)

This script uses the official HuggingFace Diffusers train_dreambooth.py approach.
Reference: https://huggingface.co/docs/diffusers/en/training/dreambooth

Key memory optimizations for 16GB:
- Gradient checkpointing
- 8-bit Adam optimizer (bitsandbytes)
- Mixed precision training (fp16)
- No text encoder training (requires 24GB+)
"""

import argparse
import os
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

logger = get_logger(__name__, log_level="INFO")


# =============================================================================
# Dataset Classes (from HuggingFace official implementation)
# =============================================================================

class PromptDataset(Dataset):
    """Dataset for generating class images (prior preservation)."""
    
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}


class DreamBoothDataset(Dataset):
    """
    Dataset for DreamBooth training with prior preservation.
    Based on HuggingFace Diffusers official implementation.
    """
    
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        
        # Load instance images
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root doesn't exist: {instance_data_root}")
        
        self.instance_images_path = list(self.instance_data_root.iterdir())
        self.instance_images_path = [
            p for p in self.instance_images_path 
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        ]
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        
        # Load class images (for prior preservation)
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.class_images_path = [
                p for p in self.class_images_path
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            ]
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
    
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        
        # Instance image
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        # Class image (if using prior preservation)
        if self.class_data_root is not None:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if class_image.mode != "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        
        return example


def collate_fn(examples, with_prior_preservation=False):
    """Collate function for DataLoader."""
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)
    
    return {"input_ids": input_ids, "pixel_values": pixel_values}


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth training script (HuggingFace style)")
    
    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    # Data
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="The prompt with identifier specifying the instance (e.g., 'a photo of sks dog')",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images for prior preservation.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt for class images (e.g., 'a photo of dog')",
    )
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        help="Flag to enable prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="Weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help="Minimal class images for prior preservation. Will generate if not enough exist.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=2,
        help="Batch size for generating class images.",
    )
    
    # Training
    parser.add_argument("--output_dir", type=str, default="dreambooth-model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Save memory at cost of speed")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam for lower memory")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Logging & Checkpointing
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=100)
    
    # SNR weighting (optional advanced feature)
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma (recommended: 5.0)")
    
    args = parser.parse_args()
    
    # Validation
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("--class_data_dir required when using --with_prior_preservation")
        if args.class_prompt is None:
            raise ValueError("--class_prompt required when using --with_prior_preservation")
    
    return args


# =============================================================================
# Class Image Generation (Prior Preservation)
# =============================================================================

def generate_class_images(args, accelerator):
    """Generate class images for prior preservation using the pretrained model."""
    class_images_dir = Path(args.class_data_dir)
    class_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Count existing images
    cur_class_images = len([
        x for x in class_images_dir.iterdir()
        if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    ])
    
    if cur_class_images < args.num_class_images:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        
        logger.info(f"Generating {args.num_class_images - cur_class_images} class images...")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=args.revision,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(accelerator.device)
        
        num_new_images = args.num_class_images - cur_class_images
        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = DataLoader(sample_dataset, batch_size=args.sample_batch_size)
        
        for example in tqdm(sample_dataloader, desc="Generating class images"):
            images = pipeline(example["prompt"]).images
            for i, image in enumerate(images):
                hash_image = hash(example["prompt"][i])
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)
        
        del pipeline
        torch.cuda.empty_cache()
        
        logger.info("✅ Class images generated!")
    else:
        logger.info(f"✅ Found {cur_class_images} existing class images, skipping generation")


# =============================================================================
# Validation (Optional)
# =============================================================================

def log_validation(pipeline, args, accelerator, epoch, global_step):
    """Generate validation images during training."""
    if args.validation_prompt is None:
        return
    
    logger.info(f"Running validation at step {global_step}...")
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(
                args.validation_prompt,
                num_inference_steps=25,
                generator=generator,
            ).images[0]
        images.append(image)
    
    # Save validation images
    val_dir = Path(args.output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        image.save(val_dir / f"step_{global_step:06d}_img_{i}.png")
    
    logger.info(f"✅ Saved {len(images)} validation images to {val_dir}")
    
    del pipeline
    torch.cuda.empty_cache()


# =============================================================================
# Compute SNR for Min-SNR Weighting (Optional)
# =============================================================================

def compute_snr(noise_scheduler, timesteps):
    """Compute SNR as per https://arxiv.org/abs/2303.09556"""
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    
    # Expand dims
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
    
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    args = parse_args()
    
    # Initialize accelerator
    logging_dir = Path(args.output_dir) / args.logging_dir
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(logging_dir)
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Generate class images if needed
    if args.with_prior_preservation:
        generate_class_images(args, accelerator)
    
    # ==========================================================================
    # Load Models
    # ==========================================================================
    
    logger.info("Loading models...")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable gradient checkpointing for memory savings
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("✅ Gradient checkpointing enabled")
    
    # Enable xformers if available
    try:
        unet.enable_xformers_memory_efficient_attention()
        logger.info("✅ xFormers memory efficient attention enabled")
    except Exception as e:
        logger.warning(f"⚠️ xFormers not available: {e}")
    
    # ==========================================================================
    # Optimizer
    # ==========================================================================
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("✅ Using 8-bit Adam optimizer")
        except ImportError:
            raise ImportError("Install bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # ==========================================================================
    # Dataset & DataLoader
    # ==========================================================================
    
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=0,  # Kaggle-safe
    )
    
    # ==========================================================================
    # Learning Rate Scheduler
    # ==========================================================================
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # ==========================================================================
    # Prepare with Accelerator
    # ==========================================================================
    
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move to device with appropriate dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("=" * 60)
    logger.info("***** Running DreamBooth Training (HuggingFace Style) *****")
    logger.info("=" * 60)
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Prior preservation = {args.with_prior_preservation}")
    logger.info(f"  Mixed precision = {args.mixed_precision}")
    logger.info(f"  Gradient checkpointing = {args.gradient_checkpointing}")
    logger.info(f"  8-bit Adam = {args.use_8bit_adam}")
    logger.info("=" * 60)
    
    global_step = 0
    
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device
                ).long()
                
                # Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Get target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")
                
                # Calculate loss
                if args.with_prior_preservation:
                    # Chunk predictions for instance and class
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    
                    # Instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # Prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    
                    # Combined loss
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Min-SNR weighting (optional)
                if args.snr_gamma is not None:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = loss * mse_loss_weights.mean()
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Checkpointing
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation
                if args.validation_prompt and global_step % args.validation_steps == 0:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=text_encoder,
                        vae=vae,
                        tokenizer=tokenizer,
                        torch_dtype=weight_dtype,
                    )
                    log_validation(pipeline, args, accelerator, epoch, global_step)
                    del pipeline
                    torch.cuda.empty_cache()
            
            if global_step >= args.max_train_steps:
                break
    
    # ==========================================================================
    # Save Final Model
    # ==========================================================================
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"✅ Model saved to {args.output_dir}")
    
    accelerator.end_training()
    logger.info("=" * 60)
    logger.info("✅ Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
