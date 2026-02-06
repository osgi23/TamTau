import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics.pairwise import cosine_similarity
import preprocessdata as pp 
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. Cấu hình GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- 2. Model và Dữ liệu ---
def get_feature_extractor():
    return ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

feature_extractor_model = get_feature_extractor()

with open("feature_data.pkl", "rb") as f:
    data = pickle.load(f)
    dataset_vectors = data["vectors"]
    dataset_paths = data["paths"]
    dataset_labels = data["labels"]

# --- 3. Hàm tìm kiếm theo Threshold ---
def find_images_by_threshold(input_image_path, threshold=0.7):
    print(f"\n--- Tìm ảnh có độ tương đồng >= {threshold} ---")
    
    input_img = pp.load_and_preprocess_image(input_image_path)
    if input_img is None: return []

    input_vector = feature_extractor_model.predict(input_img, verbose=0)
    similarities = cosine_similarity(input_vector.reshape(1, -1), dataset_vectors)[0]
    
    # Lấy tất cả các chỉ số có similarity >= threshold
    matched_indices = np.where(similarities >= threshold)[0]
    
    # Sắp xếp các kết quả tìm được theo độ giống giảm dần
    matched_indices = matched_indices[np.argsort(similarities[matched_indices])[::-1]]

    results = []
    for idx in matched_indices:
        # Bỏ qua nếu là chính nó
        if dataset_paths[idx] == input_image_path:
            continue
            
        results.append({
            "similarity": similarities[idx],
            "path": dataset_paths[idx],
            "label": dataset_labels[idx]
        })
            
    return results

# --- 4. Hàm hiển thị linh hoạt ---
def display_all_results(input_path, results):
    if not results:
        print("Không tìm thấy ảnh nào thỏa mãn ngưỡng threshold.")
        return

    n = len(results)
    print(f"Tìm thấy {n} ảnh tương đồng.")

    # Tính toán số hàng và cột cho biểu đồ (tối đa 5 cột mỗi hàng)
    cols = 5
    rows = (n + 1) // cols + (1 if (n + 1) % cols != 0 else 0)
    
    plt.figure(figsize=(20, 4 * rows))
    
    # Hiển thị ảnh gốc
    plt.subplot(rows, cols, 1)
    plt.imshow(Image.open(input_path))
    plt.title("GỐC (Input)")
    plt.axis('off')

    # Hiển thị tất cả ảnh kết quả
    for i, res in enumerate(results):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(Image.open(res['path']))
        plt.title(f"Sim: {res['similarity']:.3f}\n{res['label']}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- CHẠY THỬ ---
if __name__ == "__main__":
    test_image = r"D:\nokia.jpg"
    
    # Bạn có thể tùy chỉnh ngưỡng ở đây (Ví dụ: 0.75 hoặc 0.8)
    my_threshold = 0.6
    
    similar_images = find_images_by_threshold(test_image, threshold=my_threshold)
    
    # In danh sách đường dẫn ra terminal
    for i, res in enumerate(similar_images):
        print(f"[{i+1}] {res['similarity']:.4f} -> {res['path']}")
        
    display_all_results(test_image, similar_images)