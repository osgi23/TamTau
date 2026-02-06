import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity
import preprocessdata as pp # Để tái sử dụng hàm load_and_preprocess_image
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Cấu hình GPU (đảm bảo TensorFlow nhận GPU) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Đã kích hoạt GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

# --- 2. Tải model trích xuất đặc trưng (chỉ 1 lần) ---
def get_feature_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # pooling='avg' sẽ trực tiếp cho ra output 2048 chiều, không cần GlobalAveragePooling2D riêng nữa
    return base_model

feature_extractor_model = get_feature_extractor()

# --- 3. Tải dữ liệu vector đã trích xuất từ file (chỉ 1 lần) ---
try:
    with open("feature_data.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    dataset_vectors = loaded_data["vectors"]
    dataset_paths = loaded_data["paths"]
    dataset_labels = loaded_data["labels"]
    print(f"Đã tải {len(dataset_vectors)} vector từ feature_data.pkl.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'feature_data.pkl'.")
    print("Vui lòng chạy 'features_extractors.py' trước để tạo file dữ liệu.")
    exit() # Dừng chương trình nếu không có dữ liệu

# --- 4. Hàm tìm ảnh tương đồng ---
def find_similar_images(input_image_path, num_results=5):
    print(f"\n--- Đang tìm kiếm ảnh tương đồng với: {input_image_path} ---")
    
    # a. Tiền xử lý và trích xuất đặc trưng cho ảnh đầu vào
    input_img_preprocessed = pp.load_and_preprocess_image(input_image_path)
    
    if input_img_preprocessed is None:
        print("Lỗi: Không thể tải hoặc tiền xử lý ảnh đầu vào.")
        return []

    # predict cho ảnh input
    input_vector = feature_extractor_model.predict(input_img_preprocessed, verbose=0)
    
    # b. Tính toán Cosine Similarity
    # reshape(-1, 2048) để đảm bảo input_vector có dạng phù hợp cho cosine_similarity
    similarities = cosine_similarity(input_vector.reshape(1, -1), dataset_vectors)
    
    # c. Sắp xếp và lấy ra các ảnh tương đồng nhất
    # similarities[0] vì kết quả là một mảng 2D (1, num_images)
    sorted_indices = np.argsort(similarities[0])[::-1] # Sắp xếp giảm dần

    print(f"Các ảnh tương đồng nhất (Top {num_results}):")
    results = []
    
    # Bỏ qua chính ảnh input nếu nó có trong dataset
    count = 0
    for idx in sorted_indices:
        if dataset_paths[idx] == input_image_path:
            continue # Bỏ qua ảnh đầu vào
        
        # Chỉ lấy ảnh có điểm tương đồng > 0.7 (bạn có thể điều chỉnh)
        if similarities[0][idx] < 0.7 and count >= num_results:
            break

        print(f"  - Độ tương đồng: {similarities[0][idx]:.4f}, Đường dẫn: {dataset_paths[idx]}")
        results.append({
            "similarity": similarities[0][idx],
            "path": dataset_paths[idx]
        })
        count += 1
        if count >= num_results:
            break
            
    return results

# --- CHẠY THỬ (MAIN) ---
if __name__ == "__main__":
    # Ví dụ: Sử dụng một ảnh có sẵn trong dataset của bạn
    # Bạn có thể thay đổi đường dẫn này thành bất kỳ ảnh nào trên máy tính
    
    # Thay đổi đường dẫn này thành một ảnh CÓ SẴN trong dataset của bạn
    # Ví dụ: input_image = r"D:\Similar_Image_Finder\dataset\airplanes\image_0001.jpg"
    # Hoặc một ảnh BẤT KỲ từ bên ngoài
    
    # Ví dụ với ảnh có sẵn trong dataset của bạn
    sample_input_image = r"D:\Similar_Image_Finder\dataset\BACKGROUND_Google\image_0002.jpg"
    
    # Hoặc bạn có thể tự nhập đường dẫn ảnh từ bàn phím
    # sample_input_image = input("Nhập đường dẫn ảnh bạn muốn tìm tương đồng: ")

    similar_images = find_similar_images(sample_input_image, num_results=5)
    
    if not similar_images:
        print("Không tìm thấy ảnh tương đồng nào.")

    # Nếu bạn muốn xem ảnh trực quan, có thể dùng thư viện PIL/Pillow hoặc OpenCV

    if similar_images:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, num_results + 1, 1)
        plt.imshow(Image.open(sample_input_image))
        plt.title("Ảnh Input")
        plt.axis('off')

        for i, result in enumerate(similar_images):
            plt.subplot(1, num_results + 1, i + 2)
            plt.imshow(Image.open(result['path']))
            plt.title(f"{result['similarity']:.2f}")
            plt.axis('off')
        plt.show()