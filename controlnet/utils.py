#!/usr/bin/env python3
"""
Utility functions for DreamBooth + ControlNet training.
"""

import torch
import cv2
import numpy as np


# =============================================================================
# VRAM Debug Helper
# =============================================================================
def print_vram(label=""):
    """Print current VRAM usage for debugging."""
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
    
    Args:
        image_np: RGB numpy array (H, W, 3)
        conditioning_type: Type of conditioning to extract
            - "canny": Canny edge detection
            - "hed": HED-style soft edges (approximated with dilated Canny)
            - "none": Return None (user provides conditioning separately)
    
    Returns:
        RGB numpy array of conditioning image, or None if type is "none"
    """
    if conditioning_type == "canny":
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif conditioning_type == "hed":
        # HED-style soft edges (approximated with dilated Canny)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif conditioning_type == "none":
        return None
    
    else:
        raise ValueError(
            f"Unknown conditioning_type: {conditioning_type}. "
            f"Supported types: canny, hed, none"
        )


# =============================================================================
# DataLoader Collate Function
# =============================================================================
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
