# Similar Image Finder (SIF)
Ứng dụng tìm kiếm hình ảnh tương đồng sử dụng mô hình Deep Learning **ResNet50** và thuật toán **Cosine Similarity**.

## Tính năng
- Trích xuất đặc trưng ảnh thành vector 2048 chiều.
- Hỗ trợ tăng tốc phần cứng bằng **GPU NVIDIA** (CUDA).
- Tìm kiếm ảnh theo ngưỡng (Threshold) tùy chỉnh.
- Hiển thị kết quả trực quan bằng biểu đồ.

## Yêu cầu hệ thống
- **OS:** Windows 10/11
- **GPU:** NVIDIA (Khuyên dùng RTX 30-series trở lên, Driver mới nhất)
- **Cấu phần bổ sung:** CUDA Toolkit 11.2 và cuDNN 8.1.

## ⚙️ Cài đặt
1. Cài đặt môi trường ảo bằng Conda:
   conda create -n SIF python=3.10
   conda activate SIF
2. Cài đặt các thư viện cần thiết:
   pip install -r requirements.txt

