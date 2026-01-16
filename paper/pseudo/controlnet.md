# ControlNet Training Pseudo-code

### 1. Ý tưởng cốt lõi: Locked & Trainable
ControlNet tạo ra một bản sao của các khối Encoder từ UNet gốc. 
- **Locked_UNet:** Giữ nguyên (freezed) để bảo tồn khả năng gen ảnh/đối tượng.
- **ControlNet (Trainable):** Bản sao để học các điều kiện mới (Pose, Depth, v.v.).

### 2. Chi tiết kiến trúc kết nối
```python
# --- TRƯỚC KHI TRAIN (Khởi tạo) ---
# sd_unet là model đã fine-tune (vd: DreamBooth)
locked_encoder = sd_unet.encoder.lock() 
locked_middle  = sd_unet.middle_block.lock()
locked_decoder = sd_unet.decoder.lock()

# ControlNet là bản sao của Encoder và Middle
trainable_copy_encoder = clone(sd_unet.encoder)
trainable_copy_middle  = clone(sd_unet.middle_block)

# Các lớp Zero Convolution (Khởi tạo bằng 0)
# Giúp model ban đầu chạy y hệt bản gốc, tránh nhiễu lúc mới train.
zero_convs = [ZeroConv1x1() for _ in range(13)] # 12 cho encoder, 1 cho middle

# --- QUÁ TRÌNH FORWARD (Dự đoán nhiễu) ---
def forward(z_t, t, prompt_embeds, pose_img):
    # Biến ảnh pose (512x512) thành feature vector (64x64)
    c_f = ConditionEncoder(pose_img) 
    
    # 1. Nhánh ControlNet (Trích xuất đặc trưng tư thế)
    feat = z_t + c_f
    ctrl_outs = []
    for block in trainable_copy_encoder:
        feat = block(feat, t, prompt_embeds)
        ctrl_outs.append(feat) 
    ctrl_mid = trainable_copy_middle(feat, t, prompt_embeds)
    
    # 2. Nhánh UNet gốc (Tạo ảnh)
    # Encoder gốc
    sd_feats = []
    x = z_t
    for block in locked_encoder:
        x = block(x, t, prompt_embeds)
        sd_feats.append(x)
        
    # Middle gốc + Gợi ý từ ControlNet (qua ZeroConv)
    hidden = locked_middle(x, t, prompt_embeds) + zero_convs[12](ctrl_mid)
    
    # Decoder gốc: Nhận tín hiệu kết hợp
    # "Hãy vẽ đối tượng này (sd_feat) nhưng ở tư thế này (ctrl_out)"
    for i, block in enumerate(locked_decoder):
        skip_conn = sd_feats[i] + zero_convs[i](ctrl_outs[i])
        hidden = block(hidden, skip_conn, t, prompt_embeds)
        
    return hidden # ε_pred
```

### 3. Training Loop
```python
x0 ~ real_images           # Ảnh thực tế
pose_img ~ condition_images# Ảnh tư thế (skeleton/pose)
p ~ text_prompts          # Mô tả

# Diffusion steps
z0 = VAE.encode(x0)
t ~ Uniform(0, T)
ε ~ N(0, I)
z_t = add_noise(z0, ε, t)

# Forward pass qua hệ thống kép
c_t = TextEncoder(p)
if random() < 0.5: c_t = TextEncoder("") # Prompt dropping

ε_pred = forward(z_t, t, c_t, pose_img)

# Chỉ cập nhật ControlNet và các ZeroConvs
loss = ||ε_pred - ε||²
opt.zero_grad()
loss.backward()
opt.step()
```
