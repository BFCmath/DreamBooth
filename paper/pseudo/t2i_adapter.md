# T2I-Adapter: Architectural and Training Pseudocode

## 1. Core Concept: Lightweight Adapters
T2I-Adapter is a highly efficient, lightweight plugin (~77M parameters) designed to align external control information with the internal knowledge of a pre-trained Stable Diffusion model.

*   **Objective:** To provide spatial guidance (e.g., sketches, depth, or pose) without modifying the original U-Net weights.
*   **Integration Point:** Features are injected exclusively into the **Encoder** blocks of the U-Net.

---

## 2. Architectural Implementation
The following pseudocode details the Adapter's structure and its integration into the diffusion forward pass.

```python
# --- T2I-Adapter Architecture (ℱ_AD) ---
def T2I_Adapter(condition_map):
    # Input: High-resolution condition (e.g., 512x512)
    
    # 1. Initial spatial compression via Pixel Unshuffle
    x = PixelUnshuffle(condition_map) # -> Latent resolution (e.g., 64x64)
    
    features = []
    # 2. Multi-scale feature extraction (Scales: 64x64, 32x32, 16x16, 8x8)
    for scale in range(4):
        x = Conv2D(x)
        x = ResBlock(x)
        x = ResBlock(x)
        features.append(x)
        
        if scale < 3: # Downsample for the subsequent level
            x = Downsample(x)
            
    return features # Returns 4 scale-specific control features

# --- Combined Forward Pass ---
def forward(z_t, t, prompt_embeds, condition_map):
    # Extract features from the Lightweight Adapter
    # Optimization: This can be computed once if the condition is constant.
    adapter_feats = T2I_Adapter(condition_map)
    
    # U-Net Encoder: Direct additive injection of adapter features
    # Each adapter feature is added to the corresponding U-Net scale.
    x = z_t
    scale_idx = 0
    for i, block in enumerate(sd_unet.encoder):
        x = block(x, t, prompt_embeds)
        if i in injection_indices: # Matching U-Net and Adapter scales
            x = x + adapter_feats[scale_idx]
            scale_idx += 1
            
    # U-Net Middle and Decoder blocks proceed without further modification
    noise_pred = sd_unet.middle_and_decoder(x, t, prompt_embeds)
    
    return noise_pred
```

---

## 3. Training Strategy and Cubic Sampling
T2I-Adapter utilizes **Cubic Sampling** during training to prioritize learning structural alignment during the early stages of the diffusion process (where noise is high).

```python
# Cubic Sampling favors larger values of 't' (early diffusion stages)
def cubic_sample_t():
    u = Uniform(0, 1)
    t = (1 - u**3) * T 
    return t

# --- Training Loop ---
for x0, p, condition in dataloader:
    z0 = VAE.encode(x0)
    ε  = N(0, I)
    t  = cubic_sample_t() # Strategic timestep selection
    z_t = add_noise(z0, ε, t)
    
    c_t = TextEncoder(p)
    
    # Objective Calculation
    ε_pred = forward(z_t, t, c_t, condition)
    loss = ||ε_pred - ε||²
    
    # Optimization: Update ONLY the T2I-Adapter. SD U-Net is frozen.
    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## 4. Architectural Comparison: ControlNet vs. T2I-Adapter

| Feature                  | ControlNet                                  | T2I-Adapter                        |
| :----------------------- | :------------------------------------------ | :--------------------------------- |
| **Parameter Size**       | Large (comparable to U-Net Encoder)         | Very Small (~77M parameters)       |
| **Injection Point**      | Decoder (via Skip-connections)              | Encoder Blocks                     |
| **Connection Type**      | Zero-Convolution Residuals                  | Direct Feature Addition            |
| **Inference Efficiency** | Requires ControlNet pass for every step `t` | Features can be pre-extracted once |
