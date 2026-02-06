import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import preprocessdata as pp  
import os
import pickle
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
    image_paths = pp.get_image_paths(dataset_path)
    if len(image_paths) == 0: return np.array([]), np.array([]), np.array([])

    model = get_feature_extractor()
    
    features_list = []
    valid_paths = [] 
    labels_list = []
    
    # Gom ảnh vào một danh sách tạm
    temp_images = []
    
    print(f"Bắt đầu trích xuất bằng GPU cho {len(image_paths)} ảnh...")
    
    for idx, path in enumerate(image_paths):
        img_pre = pp.load_and_preprocess_image(path)
        if img_pre is not None:
            temp_images.append(img_pre[0]) # Lấy array ảnh (224, 224, 3)
            valid_paths.append(path)
            labels_list.append(os.path.basename(os.path.dirname(path)))
        
        # Khi đủ một "Batch" (ví dụ 32 ảnh) thì đẩy lên GPU một lần
        if len(temp_images) == 128 or (idx == len(image_paths) - 1 and len(temp_images) > 0):
            batch_array = np.array(temp_images)
            # GPU xử lý song song cả batch ở đây
            batch_features = model.predict(batch_array, verbose=0)
            features_list.append(batch_features)
            temp_images = [] # Reset batch tạm
            
            if (idx + 1) % 64 == 0 or idx == len(image_paths) - 1:
                print(f"-> Đã xử lý: {idx + 1}/{len(image_paths)} ảnh...")

    return np.vstack(features_list), np.array(valid_paths), np.array(labels_list)
# ouput của resnet50 là 7x7x2048 sau khi qua globalaveragepooling2d sẽ thành 2048 sau đó flatten thành 2048
# --- CHẠY THỬ (TEST) ---
import pickle

# --- SAU KHI CHẠY XONG HÀM TRÍCH XUẤT ---
if __name__ == "__main__":
    dataset_folder = r"D:\Similar_Image_Finder\dataset"
    vectors, paths, labels = extract_features_in_memory(dataset_folder)

    if len(vectors) > 0:
        # Lưu vào file pickle
        data_to_save = {
            "vectors": vectors,
            "paths": paths,
            "labels": labels
        }
        with open("feature_data.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
        print("--- Đã lưu dữ liệu vào file feature_data.pkl ---")