import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import preprocessdata as pp  
import os
# 1. Khởi tạo model ResNet50
def get_feature_extractor():
    # include_top=False: Bỏ lớp phân loại, chỉ lấy khung xương để trích xuất đặc trưng
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

# 2. Hàm trích xuất 
def extract_features_in_memory(dataset_path):
    # Lấy danh sách ảnh
    image_paths = pp.get_image_paths(dataset_path)
    
    model = get_feature_extractor()
    
    features_list = []
    valid_paths = [] 
    labels_list = []
    print(f"Bắt đầu trích xuất đặc trưng cho {len(image_paths)} ảnh...")
    
    for idx, path in enumerate(image_paths):
        # Xử lý ảnh
        label = os.path.basename(os.path.dirname(path))
        img_preprocessed = pp.load_and_preprocess_image(path)
        
        if img_preprocessed is not None:
            feature = model.predict(img_preprocessed, verbose=0)  
            # Flatten
            features_list.append(feature.flatten())
            valid_paths.append(path)
            labels_list.append(label)
        # In tiến độ 
        if (idx + 1) % 50 == 0:
            print(f"-> Đã xong {idx + 1} ảnh...")

    return np.array(features_list), np.array(valid_paths), np.array(labels_list)
# ouput của resnet50 là 7x7x2048 sau khi qua globalaveragepooling2d sẽ thành 2048 sau đó flatten thành 2048
# --- CHẠY THỬ (TEST) ---
if __name__ == "__main__":
    dataset_folder = r"D:\duan1\TamTau\dataset"
    
    # Gọi hàm và hứng lấy dữ liệu vào biến
    vectors, paths,labels = extract_features_in_memory(dataset_folder)
    
    # Kiểm tra nhanh
    if len(vectors) > 0:
        print(f"Vector đầu tiên có {len(vectors[0])} chiều.") # Kết quả phải là 2048
        print(f" - Tên : {labels[0]}")