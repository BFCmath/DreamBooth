x_i ~ instance_images       # "a [V] dog" (Ảnh đối tượng cần học)
x_c ~ class_images          # "a dog"     (Ảnh chung của lớp đối tượng)

x = concat(x_i, x_c)        # Gộp ảnh lại thành một batch
p = concat(p_i, p_c)        # Gộp prompt tương ứng

z0 = VAE.encode(x)          # Mã hóa ảnh sang không gian tiềm ẩn (Latent space)
c  = TextEncoder(p)         # Mã hóa văn bản (Conditioning)

t ~ Uniform(0, T)           # Chọn bước thời gian ngẫu nhiên
ε ~ N(0, I)                 # Tạo nhiễu Gaussian

z_t = add_noise(z0, ε, t)   # Thêm nhiễu vào latent
pred = UNet(z_t, t, c)      # UNet dự đoán nhiễu

target = ε

pred_i, pred_c = split(pred)    # Tách kết quả dự đoán (instance vs class)
tgt_i,  tgt_c  = split(target)  # Tách mục tiêu (instance vs class)

loss = ||pred_i - tgt_i||² + λ ||pred_c - tgt_c||²

opt.zero_grad()
loss.backward()
opt.step()