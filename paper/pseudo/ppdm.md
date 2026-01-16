x0 ~ p_data
t ~ Uniform(0, T)
ε ~ N(0, I)

x_t = q_sample(x0, t, ε)
ε^ = UNet(x_t, t)

loss = ||ε^ - ε||²

opt.zero_grad()
loss.backward()
opt.step()