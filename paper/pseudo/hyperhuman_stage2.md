# HyperHuman Stage 2: Structure-Guided Refiner Pseudocode

The second stage utilizes a pre-trained SDXL backbone to render high-resolution (1024x1024) images, guided by the structural maps generated in Stage 1.

---

## 1. Training Pipeline: High-Resolution Refinement
The goal is to fine-tune the model to utilize multi-modal structural guidance (Depth, Normal, and Pose) for hyper-realistic human synthesis.

```python
# --- 1. High-Resolution Input Sampling ---
# x_high: 1024x1024 RGB image, d, n, p: Ground truth structural maps
x_high, c, d, n, p ~ HighRes_Human_Dataset 

# --- 2. Latent Encoding (SDXL VAE) ---
z0 = VAE_XL.encode(x_high)
cond_text = TextEncoder_XL(c)

# --- 3. Robust Conditioning (Structural Dropout) ---
# Training with random dropout makes the refiner resilient to potential
# prediction errors from the Stage 1 structural model.
if random < 0.15: cond_text = ""
if random < 0.5:  d, n, p = zeros()

# --- 4. Structure Encoding (ConditionEncoder) ---
# Employs 4 convolutional layers (4x4 kernels, stride 2, ReLU) for each modality
f_d = CondEncoder_D(d) # Downsample 1024 -> 128 (latent resolution)
f_n = CondEncoder_N(n)
f_p = CondEncoder_P(p)

# --- 5. Fusion: Coordinate-wise Summation ---
# Condenses disparate structural signals into a unified control embedding
f_cond = f_d + f_n + f_p

# --- 6. Diffusion Parameters ---
t ~ Uniform(0, T)
ε ~ N(0, I)
z_t = q_sample(z0, t, ε)

# --- 7. Model Forward Pass (Frozen SDXL + Trainable Branch) ---
# f_cond is injected into trainable clones of the SDXL Encoder blocks
ε_hat = SDXL_Refiner(z_t, t, cond_text, f_cond)

# --- 8. Optimization Strategy ---
# Only the Trainable Branch and ConditionEncoders are updated.
# The SDXL backbone remains frozen to preserve large-scale generative knowledge.
loss = ||ε_hat - ε||²

opt.zero_grad()
loss.backward()
opt.step()
```

---

## 2. Inference: Sequential Generation
The finalized generation process cascades the structural predictions from Stage 1 into the high-resolution refiner.

```python
# --- Step 1: Generate structural maps via Stage 1 Model ---
# Input: User-defined Prompt and Pose
_, d_hat, n_hat = Stage1_Model(User_Prompt, User_Pose)

# --- Step 2: High-Resolution Rendering via Stage 2 Model ---
# Input: User_Prompt, User_Pose, and predicted d_hat, n_hat
t_steps = T ... 0
z_t = N(0, I)

for t in t_steps:
    # Combine user pose with predicted depth and normal maps
    f_cond = CondEncoder(User_Pose) + CondEncoder(d_hat) + CondEncoder(n_hat)
    ε_hat = SDXL_Refiner(z_t, t, TextEncoder(User_Prompt), f_cond)
    z_t = step(z_t, ε_hat, t)

# Output: Final 1024x1024 Hyper-Realistic Image
return VAE_XL.decode(z_t)
```
