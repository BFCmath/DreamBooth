# HyperHuman: tóm tắt Input/Output & Quy trình

Mô hình này chia làm 2 giai đoạn (Stages).

---

## GIAI ĐOẠN 1: Latent Structural Diffusion (G1)
**Mục tiêu:** Từ cái "Xương" (Pose), mô hình tự suy luận ra "Khối" (Depth, Normal).

### 1. Training (Lúc học)
*   **Input:** Text + Pose + (Ảnh RGB thật, Depth thật, Normal thật).
*   **Học cái gì:** Model học cách dự đoán đồng thời cả 3 bản đồ (RGB, Depth, Normal) từ Pose.
*   **Kết quả:** Một bộ não UNet hiểu được: "Nếu có cái xương này, thì chiều sâu (Depth) và hướng bề mặt (Normal) phải như thế này".

### 2. Inference (Lúc sử dụng)
*   **Input:** **Text + Pose**.
*   **Output:** **3 bản đồ dự đoán**: RGB nhòe ($\hat{x}$), Depth ($\hat{d}$), Normal ($\hat{n}$).
*   **Tại sao có được Depth & Normal?** Vì model đã là "Joint-Denoising", nó tự "tưởng tượng" ra khối 3D từ cái xương mà bạn đưa vào.

---

## GIAI ĐOẠN 2: Structure-Guided Refiner (G2)
**Mục tiêu:** Dùng "Khối" từ Stage 1 để vẽ ảnh 1024px siêu nét.

### 1. Training (Lúc học)
*   **Input:** Text + Ảnh 1024px thật + (Pose, Depth, Normal từ ảnh thật).
*   **Học cái gì:** Học cách dùng các bản đồ cấu trúc làm "khuôn" để tô màu siêu nét bằng SDXL.

### 2. Inference (Lúc sử dụng)
*   **Input:** 
    1. **Text + Pose** (Bạn nhập).
    2. **Depth ($\hat{d}$) + Normal ($\hat{n}$)** (Lấy từ kết quả của Giai đoạn 1 ở trên).
*   **Process:** Gộp 3 cái (Pose + Depth + Normal) lại thành 1 tín hiệu điều khiển SDXL.
*   **Output:** **Ảnh 1024x1024 hoàn chỉnh cuối cùng.**

---

## Tóm gọn cho Unitree Robot:
1.  Bạn đưa **Pose Robot**.
2.  **Stage 1** vẽ cho bạn cái **Depth/Normal** của con Robot đó (dù bạn không có sẵn).
3.  **Stage 2** lấy cái Depth/Normal đó để vẽ ra con **Robot Unitree** thật và nét căng.
