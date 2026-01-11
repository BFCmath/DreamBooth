#!/usr/bin/env python3
"""
DreamBooth Training Script for Stable Diffusion v1.5
This script implements proper DreamBooth with prior preservation.
Designed to run on Kaggle with GPU support.
"""

import argparse
import os
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
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
from tqdm.auto import tqdm
from PIL import Image

logger = get_logger(__name__, log_level="INFO")


class PromptDataset(Dataset):
    """Dataset for generating class images."""
    
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothDataset(Dataset):
    """
    Dataset for DreamBooth training with prior preservation.
    """
    
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        repeats=1,
    ):
        self.size = size
        self.tokenizer = tokenizer
        
        # Instance images
        self.instance_images_path = []
        for file_path in Path(instance_data_root).iterdir():
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                self.instance_images_path.append(file_path)
        
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images * repeats
        
        # Class images (for prior preservation)
        if class_data_root is not None:
            self.class_images_path = []
            for file_path in Path(class_data_root).iterdir():
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.class_images_path.append(file_path)
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
            self.class_prompt = class_prompt
        else:
            self.class_images_path = None
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        
        # Instance image
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
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
        if self.class_images_path is not None:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
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
    
    # Concat class and instance examples if using prior preservation
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.cat(input_ids, dim=0)
    
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth training script with prior preservation.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
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
        help="The prompt to specify images in the same class as the instance images (e.g., 'a photo of dog')",
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
        help="Minimal class images for prior preservation loss. Will generate if not enough exist.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size for sampling class images.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory at the cost of speed.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use.',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory for the tensorboard logs.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the training dataset.",
    )
    
    args = parser.parse_args()
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    if args.instance_data_dir is None:
        raise ValueError("You must specify --instance_data_dir")
    
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify --class_data_dir when using --with_prior_preservation")
        if args.class_prompt is None:
            raise ValueError("You must specify --class_prompt when using --with_prior_preservation")
    
    return args


def generate_class_images(args, pipeline, accelerator):
    """Generate class images for prior preservation."""
    class_images_dir = Path(args.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Count existing class images
    cur_class_images = len([x for x in class_images_dir.iterdir() if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']])
    
    if cur_class_images < args.num_class_images:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        logger.info(f"⏳ Loading pipeline to device: {accelerator.device} with dtype: {torch_dtype}")
        pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
        pipeline.set_progress_bar_config(disable=True)
        
        num_new_images = args.num_class_images - cur_class_images
        logger.info(f"📸 Need to generate {num_new_images} class images (already have {cur_class_images})")
        logger.info(f"⏳ Generating with prompt: '{args.class_prompt}'")
        
        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
        
        generated_count = 0
        for example in tqdm(
            sample_dataloader, 
            desc="Generating class images",
            disable=not accelerator.is_local_main_process,
        ):
            images = pipeline(example["prompt"], num_inference_steps=25).images
            
            for i, image in enumerate(images):
                hash_image = hash(example["prompt"][i])
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)
                generated_count += 1
            
            if generated_count % 20 == 0:
                logger.info(f"📸 Generated {generated_count}/{num_new_images} class images...")
        
        logger.info(f"✅ All {num_new_images} class images generated successfully")
        
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info(f"✅ Found {cur_class_images} existing class images (needed {args.num_class_images}), skipping generation")


def main():
    args = parse_args()
    
    # Initialize accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # Make one log on every process
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"✅ Set random seed to {args.seed}")
    
    # Generate class images if needed
    if args.with_prior_preservation:
        logger.info("=" * 60)
        logger.info("PHASE 1/6: Generating class images for prior preservation")
        logger.info("=" * 60)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
        )
        generate_class_images(args, pipeline, accelerator)
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ Class images generated and pipeline cleared from memory")
    
    # Load tokenizer and text encoder
    logger.info("=" * 60)
    logger.info("PHASE 2/6: Loading models from Hugging Face")
    logger.info("=" * 60)
    logger.info("⏳ Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    logger.info("✅ Tokenizer loaded")
    
    logger.info("⏳ Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    logger.info("✅ Text encoder loaded")
    
    # Load scheduler and models
    logger.info("⏳ Loading noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    logger.info("✅ Noise scheduler loaded")
    
    logger.info("⏳ Loading UNet (this is the largest model, may take a moment)...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    logger.info("✅ UNet loaded")
    
    logger.info("⏳ Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    logger.info("✅ VAE loaded")
    
    # Freeze vae and text_encoder
    logger.info("⏳ Freezing VAE and text encoder parameters...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    logger.info("✅ Models frozen (only UNet will be trained)")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("✅ Enabled UNet gradient checkpointing")
    
    # Enable xformers memory efficient attention if available
    try:
        unet.enable_xformers_memory_efficient_attention()
        logger.info("✅ Enabled xformers memory efficient attention")
    except Exception as e:
        logger.warning(f"⚠️  Could not enable xformers: {e}")
    
    # Prepare dataset
    logger.info("=" * 60)
    logger.info("PHASE 3/6: Preparing training dataset")
    logger.info("=" * 60)
    logger.info("⏳ Creating DreamBooth dataset...")
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt if args.with_prior_preservation else None,
        size=args.resolution,
        repeats=args.repeats,
    )
    logger.info(f"✅ Dataset created with {len(train_dataset)} total examples")
    
    def collate_fn_wrapper(examples):
        return collate_fn(examples, with_prior_preservation=args.with_prior_preservation)
    
    logger.info("⏳ Creating data loader...")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_wrapper,
        num_workers=0,  # Safer for Kaggle
    )
    logger.info(f"✅ Data loader created with {len(train_dataloader)} batches per epoch")
    
    # Prepare optimizer
    logger.info("=" * 60)
    logger.info("PHASE 4/6: Setting up optimizer and scheduler")
    logger.info("=" * 60)
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("⏳ Using 8-bit Adam optimizer...")
        except ImportError:
            raise ImportError("bitsandbytes is not installed. Install it with: pip install bitsandbytes")
    else:
        optimizer_class = torch.optim.AdamW
        logger.info("⏳ Using standard AdamW optimizer...")
    
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info(f"✅ Optimizer created (lr={args.learning_rate})")
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Prepare lr scheduler
    logger.info(f"⏳ Creating {args.lr_scheduler} learning rate scheduler...")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    logger.info("✅ LR scheduler created")
    
    # Prepare everything with accelerator
    logger.info("=" * 60)
    logger.info("PHASE 5/6: Preparing models with Accelerate")
    logger.info("=" * 60)
    logger.info("⏳ Wrapping models with Accelerate (for distributed training)...")
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("✅ Models wrapped with Accelerate")
    
    # Move vae and text_encoder to device
    logger.info(f"⏳ Moving VAE and text encoder to device: {accelerator.device}...")
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # For mixed precision training, cast vae and text_encoder to appropriate dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.info("⏳ Casting models to fp16 for mixed precision training...")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.info("⏳ Casting models to bf16 for mixed precision training...")
    else:
        logger.info("⏳ Using fp32 (no mixed precision)...")
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"✅ All models on device with dtype={weight_dtype}")
    
    # We need to recalculate our total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Initialize trackers
    if accelerator.is_main_process:
        logger.info("⏳ Initializing tracking (tensorboard)...")
        accelerator.init_trackers("dreambooth", config=vars(args))
        logger.info("✅ Trackers initialized")
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("=" * 60)
    logger.info("PHASE 6/6: TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Prior preservation = {args.with_prior_preservation}")
    logger.info("=" * 60)
    logger.info("🚀 Starting training loop...")
    logger.info("=" * 60)
    
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps), 
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"📍 Epoch {epoch + 1}/{args.num_train_epochs} started")
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
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
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Calculate loss
                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute loss separately
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    
                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    
                    # Add the prior loss to the instance loss
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log progress every 50 steps
                if global_step % 50 == 0:
                    logger.info(f"📊 Progress: {global_step}/{args.max_train_steps} steps ({100*global_step/args.max_train_steps:.1f}%) | Loss: {loss.detach().item():.4f}")
                
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    logger.info(f"💾 Saving checkpoint at step {global_step}...")
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"✅ Checkpoint saved to {save_path}")
                    
                    # Remove old checkpoints if limit is set
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        
                        if len(checkpoints) > args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                            removing_checkpoints = checkpoints[:num_to_remove]
                            
                            logger.info(f"Removing {len(removing_checkpoints)} checkpoints")
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                import shutil
                                shutil.rmtree(removing_checkpoint)
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if accelerator.sync_gradients:
                accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                logger.info(f"✅ Reached max training steps ({args.max_train_steps})")
                break
        
        logger.info(f"✅ Epoch {epoch + 1}/{args.num_train_epochs} completed")
    
    # Create the pipeline using the trained modules and save it
    logger.info("=" * 60)
    logger.info("FINALIZING: Saving trained model")
    logger.info("=" * 60)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("⏳ Creating Stable Diffusion pipeline with trained UNet...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
        )
        logger.info("⏳ Saving pipeline to disk...")
        pipeline.save_pretrained(args.output_dir)
        logger.info("=" * 60)
        logger.info(f"✅ SUCCESS! Model saved to: {args.output_dir}")
        logger.info("=" * 60)
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
