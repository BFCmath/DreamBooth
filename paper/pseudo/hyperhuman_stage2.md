# HyperHuman Stage 2: Training Pseudo-code (PPDM Style)

---

### Training Pipeline (Stage 2: Structure-Guided Refiner)
**Mục tiêu:** Fine-tune SDXL để render ảnh 1024x1024 cực nét dựa trên khung cấu trúc.

```python
# 1. High-Res Input Sampling
x_high, c, d, n, p ~ HighRes_Human_Dataset # 1024x1024

# 2. Latent Encoding (SDXL VAE)
z0 = VAE_XL.encode(x_high)
cond_text = TextEncoder_XL(c)

# 3. Robust Conditioning (Random Dropout)
# Giúp model không bị "lệch tủ" nếu G1 dự đoán Depth/Normal hơi lỗi
if random < 0.15: cond_text = ""
if random < 0.5:  d, n, p = zeros()

# 4. Structure Encoding (ConditionEncoder)
# 4 Conv layers (4x4, stride 2, ReLU) cho từng loại
f_d = CondEncoder_D(d) # 1024 -> 128
f_n = CondEncoder_N(n)
f_p = CondEncoder_P(p)

# 5. Fusion: Coordinate-wise Summation
# Nén 3 loại cấu trúc thành 1 tín hiệu điều khiển duy nhất
f_cond = f_d + f_n + f_p

# 6. Diffusion Parameters
t ~ Uniform(0, T)
ε ~ N(0, I)
z_t = q_sample(z0, t, ε)

# 7. Model Forward Pass (Frozen SDXL + Trainable Copies)
# f_cond được nạp vào các bản sao (Trainable copies) của SDXL Encoder
ε_hat = SDXL_Refiner(z_t, t, cond_text, f_cond)

# 8. Loss & Optimization
# CHỈ update Trainable Copies và CondEncoders. Frozen SDXL backbone.
loss = ||ε_hat - ε||²

opt.zero_grad()
loss.backward()
opt.step()
```

---

### Inference (Lúc sử dụng)
```python
# 1. Lấy cấu trúc từ Stage 1
# Input: User_Prompt, User_Pose
_, d_hat, n_hat = Stage1_Model(User_Prompt, User_Pose)

# 2. Render ảnh nét cao ở Stage 2
# Input: User_Prompt, User_Pose, d_hat, n_hat
t_steps = T ... 0
z_t = N(0, I)

for t in t_steps:
    f_cond = CondEncoder(User_Pose) + CondEncoder(d_hat) + CondEncoder(n_hat)
    ε_hat = SDXL_Refiner(z_t, t, TextEncoder(User_Prompt), f_cond)
    z_t = step(z_t, ε_hat, t)

# Output: Final 1024x1024 Hyper-Realistic Image
return VAE_XL.decode(z_t)
```
