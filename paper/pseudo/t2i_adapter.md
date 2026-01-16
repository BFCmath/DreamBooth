# T2I-Adapter Training Pseudo-code

### 1. Ý tưởng cốt lõi: Lightweight Adapter
T2I-Adapter là một mạng cực kỳ nhỏ (~77M tham số) tách biệt hoàn toàn với UNet gốc. 
- **Mục tiêu:** "Căn chỉnh" (align) thông tin điều kiện bên ngoài với kiến thức bên trong model SD.
- **Vị trí tác động:** Chỉ thêm vào nhánh **Encoder** của UNet.

### 2. Chi tiết kiến trúc Adapter
```python
# --- KIẾN TRÚC ADAPTER (ℱ_AD) ---
def T2I_Adapter(condition_img):
    # condition_img: 512x512
    
    # 1. Downsample ban đầu bằng Pixel Unshuffle
    x = PixelUnshuffle(condition_img) # -> 64x64
    
    features = []
    # 2. Qua 4 level độ phân giải (64x64, 32x32, 16x16, 8x8)
    for scale in range(4):
        x = Conv2D(x)
        x = ResBlock(x)
        x = ResBlock(x)
        features.append(x)
        
        if scale < 3: # Downsample cho level tiếp theo
            x = Downsample(x)
            
    return features # Trả về 4 feature maps tương ứng 4 level của UNet Encoder

# --- QUÁ TRÌNH FORWARD ---
def forward(z_t, t, prompt_embeds, condition_img):
    # Trích xuất đặc trưng từ Adapter (Chỉ chạy 1 lần nếu condition không đổi)
    adapter_feats = T2I_Adapter(condition_img)
    
    # UNet Encoder: Cộng trực tiếp các đặc trưng từ adapter
    # i đại diện cho 4 scale: 64x64, 32x32, 16x16, 8x8
    x = z_t
    for i, block in enumerate(sd_unet.encoder):
        x = block(x, t, prompt_embeds)
        if i in injection_indices: # Tại mỗi scale của encoder
            x = x + adapter_feats[scale_idx]
            scale_idx += 1
            
    # UNet Middle & Decoder chạy bình thường (không có can thiệp thêm)
    noise_pred = sd_unet.middle_and_decoder(x, t, prompt_embeds)
    
    return noise_pred
```

### 3. Training Loop & Chiến thuật Cubic Sampling
```python
# T2I-Adapter dùng Cubic Sampling để model tập trung học cấu trúc ở các bước nhiễu nặng
def cubic_sample_t():
    u = Uniform(0, 1)
    t = (1 - u**3) * T # Hàm bậc 3 giúp lấy t lớn (early stages) nhiều hơn
    return t

for x0, p, condition in dataloader:
    z0 = VAE.encode(x0)
    ε  = N(0, I)
    t  = cubic_sample_t() # Điểm khác biệt quan trọng
    z_t = add_noise(z0, ε, t)
    
    c_t = TextEncoder(p)
    
    # Forward & Loss
    ε_pred = forward(z_t, t, c_t, condition)
    loss = ||ε_pred - ε||²
    
    # CHỈ update T2I-Adapter, SD_UNet bị đóng băng hoàn toàn
    opt.zero_grad()
    loss.backward()
    opt.step()
```

### 4. So sánh nhanh với ControlNet:
| Đặc điểm | ControlNet | T2I-Adapter |
| :--- | :--- | :--- |
| **Kích thước** | Lớn (bằng cả Encoder UNet) | Rất nhỏ (~77M params) |
| **Vị trí Inject** | Decoder (Skip-connections) | Encoder |
| **Kết nối** | Zero Convolution | Cộng trực tiếp |
| **Inference** | Phải chạy ControlNet mỗi bước `t` | Có thể trích xuất feature 1 lần duy nhất |
