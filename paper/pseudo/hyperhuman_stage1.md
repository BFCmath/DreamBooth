# HyperHuman Stage 1: Structural Learning Pseudocode

The first stage focuses on learning joint structural representations (RGB, Depth, and Normal maps) from human pose data via a multi-task learning framework.

---

## 1. Training Pipeline: Latent Structural Diffusion
In this stage, the model learns to synthesize fine-grained structural maps that are consistent with a provided skeleton pose.

```python
# --- 1. Input Sampling ---
# x: RGB image, d: Depth map, n: Normal map, c: Text prompt, p: Pose image
x, d, n, c, p ~ HumanVerse_Dataset

# --- 2. Latent Encoding ---
z0_x = VAE.encode(x)
z0_d = VAE.encode(d)
z0_n = VAE.encode(n)
cond = TextEncoder(c)

# --- 3. Diffusion Parameters ---
t ~ Uniform(0, T)
ε_x, ε_d, ε_n ~ N(0, I)

# --- 4. Forward Diffusion (v-prediction schedule) ---
# z_t = α_t * z0 + σ_t * ε
z_t_x = q_sample(z0_x, t, ε_x)
z_t_d = q_sample(z0_d, t, ε_d)
z_t_n = q_sample(z0_n, t, ε_n)

# --- 5. Forward Pass: Expert + Shared UNet Architecture ---
# 5.1 Expert Encoders: Mapping distinct modalities to a common latent distribution
f_x = Expert_Down_X(z_t_x, t, cond, p)
f_d = Expert_Down_D(z_t_d, t, cond, p)
f_n = Expert_Down_N(z_t_n, t, cond, p)

# 5.2 Latent Fusion: Element-wise summation of modality features
f_fused = f_x + f_d + f_n

# 5.3 Shared Backbone: Processing fused structural information
f_shared = Shared_Backbone(f_fused, t, cond, p)

# 5.4 Expert Decoders: Modality-specific structural prediction
v_hat_x = Expert_Up_X(f_shared)
v_hat_d = Expert_Up_D(f_shared)
v_hat_n = Expert_Up_N(f_shared)

# --- 6. Optimization Targets (v-prediction) ---
# v = α_t * ε - σ_t * z0
v_t_x = α_t * ε_x - σ_t * z0_x
v_t_d = α_t * ε_d - σ_t * z0_d
v_t_n = α_t * ε_n - σ_t * z0_n

# --- 7. Multi-task Loss Calculation ---
loss = ||v_hat_x - v_t_x||² + ||v_hat_d - v_t_d||² + ||v_hat_n - v_t_n||²

opt.zero_grad()
loss.backward()
opt.step()
```

---

## 2. Inference: Joint Structural Generation
During inference, the model generates all structural maps simultaneously from a text prompt and pose input.

```python
# Input: Pose (p) and Text Condition (cond)
t_steps = T ... 0
z_t_x, z_t_d, z_t_n ~ N(0, I)

for t in t_steps:
    # Model predicts noise parameters for all three modalities jointly
    v_hat_x, v_hat_d, v_hat_n = Stage1_UNet(z_t_x, z_t_d, z_t_n, t, cond, p)
    
    # Update latents for each modality
    z_t_x = step(z_t_x, v_hat_x, t)
    z_t_d = step(z_t_d, v_hat_d, t)
    z_t_n = step(z_t_n, v_hat_n, t)

# Output: Synthesized RGB (low-res), Depth, and Normal maps
return VAE.decode(z_t_x), VAE.decode(z_t_d), VAE.decode(z_t_n)
```
