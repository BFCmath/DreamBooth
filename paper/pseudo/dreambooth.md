# DreamBooth: Personalized Diffusion Training Pseudocode

This pseudocode illustrates the prioritized fine-tuning process of DreamBooth, which incorporates **Class-Preservation Loss** to prevent catastrophic forgetting.

---

## Training Logic

```python
# --- Input Preparation ---
x_i ~ instance_images       # Images of the specific subject (e.g., "[V] dog")
x_c ~ class_images          # Generic images of the same class (e.g., "dog")

x = concat(x_i, x_c)        # Combined batch for regularization
p = concat(p_i, p_c)        # Corresponding prompts (Instance and Class descriptors)

# --- Encoding Phase ---
z0 = VAE.encode(x)          # Project images to latent space
c  = TextEncoder(p)         # Textual conditioning embeddings

# --- Diffusion Forward Process ---
t ~ Uniform(0, T)           # Random timestep sampling
ε ~ N(0, I)                 # Gaussian noise generation
z_t = add_noise(z0, ε, t)   # Diffuse the latent representations

# --- Model Inference ---
pred = UNet(z_t, t, c)      # U-Net noise prediction
target = ε

# --- Multi-task Optimization ---
# Splitting predictions to calculate instance and regularization losses separately
pred_i, pred_c = split(pred)    
tgt_i,  tgt_c  = split(target)  

# Combine reconstruction loss with Prior Preservation Loss (λ weight)
loss = ||pred_i - tgt_i||² + λ ||pred_c - tgt_c||²

# --- Gradient Update ---
opt.zero_grad()
loss.backward()
opt.step()
```