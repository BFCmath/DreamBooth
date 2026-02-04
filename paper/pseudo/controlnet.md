# ControlNet: Architectural and Training Pseudocode

## 1. Core Concept: Locked and Trainable Branches
ControlNet extends a pre-trained Stable Diffusion model by creating a trainable clone of its encoder blocks.

*   **Locked U-Net:** The original U-Net blocks (Encoder, Middle, and Decoder) are frozen to preserve the foundational image generation and semantic knowledge.
*   **Trainable ControlNet Branch:** A clone of the Encoder and Middle blocks that learns to process external condition maps (e.g., Canny edges, Human Pose, Depth maps).

---

## 2. Architectural Implementation
The following pseudocode outlines the initialization and forward pass of the ControlNet system.

```python
# --- Initialization Phase ---
# 'sd_unet' refers to the pre-trained/fine-tuned model (e.g., via DreamBooth)
locked_encoder = sd_unet.encoder.lock() 
locked_middle  = sd_unet.middle_block.lock()
locked_decoder = sd_unet.decoder.lock()

# ControlNet is a trainable clone of the Encoder and Middle blocks
trainable_copy_encoder = clone(sd_unet.encoder)
trainable_copy_middle  = clone(sd_unet.middle_block)

# Zero Convolutions: Initialized to zero to ensure zero-residual impact at start
# This allows the model to output identical results to the original U-Net at step 0.
zero_convs = [ZeroConv1x1() for _ in range(13)] # 12 for encoder scales, 1 for middle

# --- Forward Pass (Noise Prediction) ---
def forward(z_t, t, prompt_embeds, condition_map):
    # Process condition map (e.g., 512x512) into a compatible latent feature
    c_f = ConditionEncoder(condition_map) 
    
    # 1. ControlNet Branch: Extract structural features
    feat = z_t + c_f
    ctrl_outs = []
    for block in trainable_copy_encoder:
        feat = block(feat, t, prompt_embeds)
        ctrl_outs.append(feat) 
    ctrl_mid = trainable_copy_middle(feat, t, prompt_embeds)
    
    # 2. Original U-Net Branch: Generate imagery with structural guidance
    sd_feats = []
    x = z_t
    for block in locked_encoder:
        x = block(x, t, prompt_embeds)
        sd_feats.append(x)
        
    # Inject ControlNet signals into the Middle Block via ZeroConv
    hidden = locked_middle(x, t, prompt_embeds) + zero_convs[12](ctrl_mid)
    
    # Inject ControlNet signals into the Decoder Blocks (Skip-Connections)
    # The Decoder receives combined guidance from SD Encoder and ControlNet
    for i, block in enumerate(locked_decoder):
        skip_conn = sd_feats[i] + zero_convs[i](ctrl_outs[i])
        hidden = block(hidden, skip_conn, t, prompt_embeds)
        
    return hidden # Final noise prediction ε_pred
```

---

## 3. Training Objective
The training focuses solely on the trainable branch and the zero-convolution layers.

```python
x0 ~ real_images            # Reference Training Sample
pose_img ~ condition_maps   # Structural condition (e.g., OpenPose)
p ~ text_prompts           # Textual description

# Latent diffusion process
z0 = VAE.encode(x0)
t ~ Uniform(0, T)
ε ~ N(0, I)
z_t = add_noise(z0, ε, t)

# Conditional preparation
c_t = TextEncoder(p)
if random() < 0.5: c_t = TextEncoder("") # Classifier-Free Guidance (CFG) Drop

# Optimization
ε_pred = forward(z_t, t, c_t, pose_img)
loss = ||ε_pred - ε||²

opt.zero_grad()
loss.backward()
opt.step() # Updates ONLY ControlNet and ZeroConv layers
```
