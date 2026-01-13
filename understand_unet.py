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
    
    # UNet exploration mode
    parser.add_argument("--explore_unet", action="store_true", help="Deep dive into UNet architecture")
    
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
# UNet Architecture Exploration
# =============================================================================

def count_parameters(module):
    """Count trainable and total parameters in a module."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def format_params(num):
    """Format parameter count in human readable form."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def explore_unet_architecture(args):
    """Deep dive into UNet2DConditionModel architecture."""
    
    print("\n" + "=" * 80)
    print("🔬 DEEP DIVE: UNet2DConditionModel Architecture Analysis")
    print("=" * 80 + "\n")
    
    # Load UNet only
    print("📦 Loading UNet from pretrained model...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    
    # =========================================================================
    # 1. High-Level Configuration
    # =========================================================================
    print("\n" + "=" * 80)
    print("📋 SECTION 1: UNet Configuration")
    print("=" * 80)
    
    config = unet.config
    print(f"""
    Model Type:          {config._class_name if hasattr(config, '_class_name') else 'UNet2DConditionModel'}
    Sample Size:         {config.sample_size}
    In Channels:         {config.in_channels}
    Out Channels:        {config.out_channels}
    Center Input Sample: {config.center_input_sample}
    
    Block Out Channels:  {config.block_out_channels}
    Down Block Types:    {config.down_block_types}
    Up Block Types:      {config.up_block_types}
    
    Layers per Block:    {config.layers_per_block}
    Attention Head Dim:  {config.attention_head_dim}
    Cross Attention Dim: {config.cross_attention_dim}
    """)
    
    # =========================================================================
    # 2. Parameter Count Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 SECTION 2: Parameter Count Summary")
    print("=" * 80)
    
    trainable, total = count_parameters(unet)
    print(f"""
    Total Parameters:     {format_params(total)} ({total:,})
    Trainable Parameters: {format_params(trainable)} ({trainable:,})
    Frozen Parameters:    {format_params(total - trainable)} ({total - trainable:,})
    """)
    
    # Parameter breakdown by major component
    print("\n    Parameter Breakdown by Component:")
    print("    " + "-" * 50)
    
    components = {
        'conv_in': unet.conv_in,
        'time_embedding': unet.time_embedding,
        'down_blocks': unet.down_blocks,
        'mid_block': unet.mid_block,
        'up_blocks': unet.up_blocks,
        'conv_norm_out': unet.conv_norm_out,
        'conv_out': unet.conv_out,
    }
    
    for name, module in components.items():
        _, params = count_parameters(module)
        percentage = (params / total) * 100
        bar = "█" * int(percentage // 2) + "░" * (50 - int(percentage // 2))
        print(f"    {name:20s}: {format_params(params):>10s} ({percentage:5.2f}%) {bar[:20]}")
    
    # =========================================================================
    # 3. Encoder (Down Blocks) Analysis
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("⬇️  SECTION 3: Encoder (Down Blocks) Analysis")
    print("=" * 80)
    
    print(f"\n    Number of Down Blocks: {len(unet.down_blocks)}\n")
    
    for i, block in enumerate(unet.down_blocks):
        block_type = type(block).__name__
        _, params = count_parameters(block)
        
        print(f"\n    ┌─ Down Block {i} ({block_type})")
        print(f"    │  Parameters: {format_params(params)}")
        
        # Check for ResNet blocks
        if hasattr(block, 'resnets'):
            print(f"    │  ResNet Blocks: {len(block.resnets)}")
            for j, resnet in enumerate(block.resnets):
                _, rp = count_parameters(resnet)
                print(f"    │    └─ ResNet[{j}]: {format_params(rp)}")
        
        # Check for Attention blocks
        if hasattr(block, 'attentions') and block.attentions is not None:
            print(f"    │  Attention Blocks: {len(block.attentions)}")
            for j, attn in enumerate(block.attentions):
                _, ap = count_parameters(attn)
                print(f"    │    └─ Attention[{j}]: {format_params(ap)}")
                # Dive into transformer blocks
                if hasattr(attn, 'transformer_blocks'):
                    for k, tb in enumerate(attn.transformer_blocks):
                        print(f"    │         └─ TransformerBlock[{k}]:")
                        if hasattr(tb, 'attn1'):
                            _, a1p = count_parameters(tb.attn1)
                            print(f"    │              └─ Self-Attention: {format_params(a1p)}")
                        if hasattr(tb, 'attn2'):
                            _, a2p = count_parameters(tb.attn2)
                            print(f"    │              └─ Cross-Attention: {format_params(a2p)}")
                        if hasattr(tb, 'ff'):
                            _, ffp = count_parameters(tb.ff)
                            print(f"    │              └─ FeedForward: {format_params(ffp)}")
        
        # Check for Downsampler
        if hasattr(block, 'downsamplers') and block.downsamplers is not None:
            print(f"    │  Downsampler: Yes")
        else:
            print(f"    │  Downsampler: No (preserves spatial size)")
        
        print(f"    └─")
    
    # =========================================================================
    # 4. Bottleneck (Mid Block) Analysis
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("🔄 SECTION 4: Bottleneck (Mid Block) Analysis")
    print("=" * 80)
    
    mid = unet.mid_block
    mid_type = type(mid).__name__
    _, mid_params = count_parameters(mid)
    
    print(f"""
    Block Type: {mid_type}
    Parameters: {format_params(mid_params)}
    """)
    
    if hasattr(mid, 'resnets'):
        print(f"    ResNet Blocks: {len(mid.resnets)}")
    if hasattr(mid, 'attentions') and mid.attentions is not None:
        print(f"    Attention Blocks: {len(mid.attentions)}")
        for j, attn in enumerate(mid.attentions):
            _, ap = count_parameters(attn)
            print(f"      └─ Attention[{j}]: {format_params(ap)}")
    
    # =========================================================================
    # 5. Decoder (Up Blocks) Analysis
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("⬆️  SECTION 5: Decoder (Up Blocks) Analysis")
    print("=" * 80)
    
    print(f"\n    Number of Up Blocks: {len(unet.up_blocks)}\n")
    
    for i, block in enumerate(unet.up_blocks):
        block_type = type(block).__name__
        _, params = count_parameters(block)
        
        print(f"\n    ┌─ Up Block {i} ({block_type})")
        print(f"    │  Parameters: {format_params(params)}")
        
        if hasattr(block, 'resnets'):
            print(f"    │  ResNet Blocks: {len(block.resnets)}")
        
        if hasattr(block, 'attentions') and block.attentions is not None:
            print(f"    │  Attention Blocks: {len(block.attentions)}")
        
        if hasattr(block, 'upsamplers') and block.upsamplers is not None:
            print(f"    │  Upsampler: Yes")
        else:
            print(f"    │  Upsampler: No")
        
        print(f"    └─")
    
    # =========================================================================
    # 6. Cross-Attention Deep Dive
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("🎯 SECTION 6: Cross-Attention Mechanism Deep Dive")
    print("=" * 80)
    
    print("""
    Cross-attention is the KEY mechanism that allows the UNet to be conditioned
    on text embeddings from CLIP. Here's how it works:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Cross-Attention Flow:                                                    │
    │                                                                          │
    │  [Image Features]      [Text Embeddings]                                │
    │        │                      │                                          │
    │        ▼                      ▼                                          │
    │   ┌─────────┐           ┌─────────┐                                     │
    │   │ Query Q │           │ Key K   │                                     │
    │   │ (from   │           │ Value V │                                     │
    │   │ image)  │           │ (from   │                                     │
    │   └────┬────┘           │ text)   │                                     │
    │        │                └────┬────┘                                     │
    │        │                     │                                          │
    │        └────────┬────────────┘                                          │
    │                 ▼                                                        │
    │         ┌──────────────┐                                                │
    │         │ Attention =  │                                                │
    │         │ softmax(QK^T)│                                                │
    │         │ / sqrt(d_k)  │                                                │
    │         └──────┬───────┘                                                │
    │                ▼                                                         │
    │         ┌──────────────┐                                                │
    │         │ Output =     │                                                │
    │         │ Attention @ V│                                                │
    │         └──────────────┘                                                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Find and print a sample cross-attention layer
    sample_attn = None
    for block in unet.down_blocks:
        if hasattr(block, 'attentions') and block.attentions is not None:
            if len(block.attentions) > 0:
                sample_attn = block.attentions[0]
                break
    
    if sample_attn is not None and hasattr(sample_attn, 'transformer_blocks'):
        tb = sample_attn.transformer_blocks[0]
        if hasattr(tb, 'attn2'):
            cross_attn = tb.attn2
            print("\n    Sample Cross-Attention Layer Details:")
            print("    " + "-" * 50)
            print(f"    Query dimension: {cross_attn.to_q.in_features} -> {cross_attn.to_q.out_features}")
            print(f"    Key dimension:   {cross_attn.to_k.in_features} -> {cross_attn.to_k.out_features}")
            print(f"    Value dimension: {cross_attn.to_v.in_features} -> {cross_attn.to_v.out_features}")
            if hasattr(cross_attn, 'heads'):
                print(f"    Number of heads: {cross_attn.heads}")
    
    # =========================================================================
    # 7. Time Embedding Analysis
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("⏰ SECTION 7: Time Embedding Analysis")
    print("=" * 80)
    
    time_emb = unet.time_embedding
    _, time_params = count_parameters(time_emb)
    
    print(f"""
    The time embedding encodes the diffusion timestep (t) into the model.
    This tells the UNet "how noisy" the input is and what to expect.
    
    Time Embedding Type: {type(time_emb).__name__}
    Parameters: {format_params(time_params)}
    """)
    
    if hasattr(time_emb, 'linear_1'):
        print(f"    Linear 1: {time_emb.linear_1.in_features} -> {time_emb.linear_1.out_features}")
    if hasattr(time_emb, 'linear_2'):
        print(f"    Linear 2: {time_emb.linear_2.in_features} -> {time_emb.linear_2.out_features}")
    
    # =========================================================================
    # 8. Tensor Shape Flow
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("📐 SECTION 8: Tensor Shape Flow Through UNet")
    print("=" * 80)
    
    print(f"""
    Input Shape Assumptions:
      - Batch size: 1
      - Latent channels: {config.in_channels}
      - Spatial size: {config.sample_size}x{config.sample_size}
      - Text embedding: (1, 77, {config.cross_attention_dim})
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Forward Pass Shape Flow:                                                 │
    │                                                                          │
    │ Input Latent: (1, {config.in_channels}, {config.sample_size}, {config.sample_size})                                     │
    │      │                                                                   │
    │      ▼ conv_in                                                          │
    │ (1, {config.block_out_channels[0]}, {config.sample_size}, {config.sample_size})                                       │
    │      │                                                                   │
    │      ▼ Down Block 0                                                      │
    │ (1, {config.block_out_channels[0]}, {config.sample_size}, {config.sample_size}) -> (1, {config.block_out_channels[0]}, {config.sample_size//2}, {config.sample_size//2})                │
    │      │                                                                   │
    │      ▼ Down Block 1                                                      │
    │ (1, {config.block_out_channels[1]}, {config.sample_size//2}, {config.sample_size//2}) -> (1, {config.block_out_channels[1]}, {config.sample_size//4}, {config.sample_size//4})                    │
    │      │                                                                   │
    │      ▼ Down Block 2                                                      │
    │ (1, {config.block_out_channels[2]}, {config.sample_size//4}, {config.sample_size//4}) -> (1, {config.block_out_channels[2]}, {config.sample_size//8}, {config.sample_size//8})                     │
    │      │                                                                   │
    │      ▼ Down Block 3 (no downsample)                                      │
    │ (1, {config.block_out_channels[3]}, {config.sample_size//8}, {config.sample_size//8})                                        │
    │      │                                                                   │
    │      ▼ Mid Block                                                         │
    │ (1, {config.block_out_channels[3]}, {config.sample_size//8}, {config.sample_size//8})                                        │
    │      │                                                                   │
    │      ▼ Up Blocks (reverse + skip connections)                            │
    │ ...gradually upsampling back to original size...                         │
    │      │                                                                   │
    │      ▼ conv_out                                                          │
    │ Output: (1, {config.out_channels}, {config.sample_size}, {config.sample_size})                                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # =========================================================================
    # 9. Full Architecture Print
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("📝 SECTION 9: Full UNet Architecture (Raw PyTorch Print)")
    print("=" * 80 + "\n")
    print(unet)
    
    # =========================================================================
    # 10. Individual Block Details
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("🔍 SECTION 10: Sample Attention Block Detail")
    print("=" * 80)
    
    if len(unet.down_blocks) > 1 and hasattr(unet.down_blocks[1], 'attentions'):
        if unet.down_blocks[1].attentions is not None:
            print("\n    Down Block 1 - Attention Block 0:")
            print("    " + "-" * 50)
            print(unet.down_blocks[1].attentions[0])
    
    print("\n\n" + "=" * 80)
    print("✅ UNet Architecture Analysis Complete!")
    print("=" * 80 + "\n")
    
    # Cleanup
    del unet
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    args = parse_args()
    
    # If explore mode, run architecture analysis and exit
    if args.explore_unet:
        explore_unet_architecture(args)
        return
    
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
    
    print("=" * 30)
    print("U-Net Architecture Overview")
    print("=" * 30)
    
    # Option 1: Print the full raw architecture
    print(unet) 
    
    # Option 2: Print a specific block to see the "Conditioning" mechanism clearly
    # This prints just the first Down-sampling block's attention layers
    print("\n--- INSIDE A CROSS-ATTENTION BLOCK ---")
    print(unet.down_blocks[1].attentions[0])


if __name__ == "__main__":
    main()
