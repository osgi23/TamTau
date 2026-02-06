# Similar Image Finder (SIF): Ứng dụng tìm kiếm hình ảnh tương đồng 

## Yêu cầu hệ thống
- **OS:** Windows 10/11
- **GPU:** NVIDIA (Khuyên dùng RTX 30-series trở lên, Driver mới nhất)
- **Cấu phần bổ sung:** CUDA Toolkit 11.2 và cuDNN 8.1.

##  Cài đặt
1. Cài đặt môi trường ảo bằng Conda:
   conda create -n SIF python=3.10
   conda activate SIF
2. Cài đặt các thư viện cần thiết:
   pip install -r requirements.txt
# Chi tiết
- sử dụng dataset caltech101 : https://www.kaggle.com/datasets/imbikramsaha/caltech-101
- 3 bước chính : preprocess data -> trích xuất đặc trưng ->tìm kiếm dựa trên thuật toán cosine similarity
+) tiền xử lý dữ liệu : resize ảnh về kích thước 224x224 -> thêm chiều dữ liệu (1,224,224,3) để phù hợp với đầu vào của resnet50
+) trích xuất đặc trưng bằng resnet50 : output sẽ là 9144 vector 2048 chiều được average pooling từ ảnh có kích cỡ 7x7x2048
+) tìm ảnh tương đồng : nếu ảnh đầu vào có độ tương đồng tính được >= threshold có thể thay đổi được thì sẽ được in ra kết quả 
-> DEMO SIMILAR_SEARCH_FINDER's impressed by face-verify from HONGSON507

Authored : TranThuPhuong111,Osgi23,HongSon507 
