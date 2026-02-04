# PPDM: Personalized Probabilistic Diffusion Models Pseudocode

This pseudocode captures the fundamental diffusion training objective used as a backbone for personalized generation.

```python
x0 ~ p_data                 # Real data sample
t ~ Uniform(0, T)           # Random timestep
ε ~ N(0, I)                 # Gaussian noise

# Diffusion forward process
x_t = q_sample(x0, t, ε)

# U-Net noise prediction
ε_hat = UNet(x_t, t)

# Simple L2 Reconstruction Loss
loss = ||ε_hat - ε||²

# Gradient step
opt.zero_grad()
loss.backward()
opt.step()
```