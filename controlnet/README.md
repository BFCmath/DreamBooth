# ControlNet & DreamBooth Integration Module

This directory contains advanced scripts for fine-tuning **ControlNet** and integrating it with **DreamBooth** for identity-consistent, pose-controlled image generation.

## 📁 Key Components

### 1. Data Preparation
Before training, you must extract structural conditioning (Pose, Canny edges, etc.) from your raw images.
*   `extract_pose.py`: Extracts OpenPose skeletons from images.
*   `extract_canny.py`: Extracts Canny edges from images.
*   `prepare_dreambooth_dataset.py`: Organizes images for combined DreamBooth/ControlNet training.

### 2. Training Workflows

#### Standard ControlNet Fine-tuning
Used to adapt a pretrained ControlNet to a specific domain or new conditioning type.
*   **Script:** `train_controlnet.py`
*   **Launcher:** `bash run_train.sh`

#### Identity-Vested ControlNet (DreamBooth Integration)
Optimizes a ControlNet branch while preserving specific object/person identity using DreamBooth rare tokens (e.g., `sks`).
*   **Script:** `dreambooth_controlnet.py`
*   **Launcher:** `bash run_dreambooth_controlnet.sh`

#### Multi-Stage Refinement (HyperHuman Style)
Advanced cascaded training for superior structural fidelity.
*   **Stage 1 (`dreambooth_controlnet_stage1.py`):** Focuses on latent structural learning and spatial alignment.
*   **Stage 2 (`dreambooth_controlnet_stage2.py`):** Focuses on high-resolution (1024px) detail refinement.

---

## 🚀 Execution Guide (Kaggle P100/T4)

### Step 1: Clone and Setup
```bash
git clone https://github.com/BFCmath/DreamBooth.git
cd DreamBooth/controlnet
pip install -r requirements.txt
```

### Step 2: Extract Conditioning
```bash
python extract_pose.py \
    --input_dir ./unitree_dataset \
    --output_dir ./data \
    --instance_prompt "a sks humanoid robot"
```

### Step 3: Run Training
The `run_dreambooth_controlnet.sh` script is the recommended entry point for identity-consistent control.
```bash
export INSTANCE_PROMPT="a sks humanoid robot"
export MAX_TRAIN_STEPS=400
bash run_dreambooth_controlnet.sh
```

### Step 4: Inference
Test your fine-tuned model using the inference suite:
```bash
python infer_controlnet.py \
    --prompt "a sks humanoid robot running in a forest" \
    --input_image ./data/conditioning/001.png \
    --controlnet_model ./output/controlnet-dreambooth \
    --output_dir ./inference_output
```

---

## 📊 Directory Structure Requirements
For most scripts, the following layout is expected:
```
data/
├── images/           # Ground truth images (RGB)
├── conditioning/     # Pre-extracted maps (Pose/Canny) - filenames must match images
└── prompts.txt       # (Optional) Detailed prompts per image
```