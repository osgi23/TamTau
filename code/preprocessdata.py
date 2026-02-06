import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Kích thước ảnh chuẩn cho ResNet50 để có thể dùng cho sprint sau
IMG_SIZE = (224, 224)

def get_image_paths(root_path):
    image_paths = []
    # Quét tất cả file trong các subfolder
    for ext in ['*.jpg']:
        pattern = os.path.join(root_path, '*', ext)
        image_paths.extend(glob.glob(pattern))
    
    print(f"Đã tìm thấy tổng cộng {len(image_paths)} ảnh.")
    return sorted(image_paths)

def load_and_preprocess_image(image_path):

    try:
        # Load ảnh + Resize
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        
        # Thêm chiều batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Chuẩn hóa theo ResNet
        img_processed = preprocess_input(img_array)
        
        return img_processed
    except Exception as e: # Nếu ảnh bị lỗi đọc hoặc xử lý, in lỗi và trả về None
        print(f"Lỗi đọc file {image_path}: {e}")
        return None

# --- CHẠY THỬ ---
if __name__ == "__main__":

    dataset_folder = r"D:\Similar_Image_Finder\dataset" 
    
    # 1. Lấy danh sách
    paths = get_image_paths(dataset_folder)
    
    # 2. Test thử ảnh đầu tiên tìm được
    if len(paths) > 0:
        print(f"\nVí dụ ảnh đầu tiên: {paths[2]}")
        # Thử xử lý ảnh đó
        img_vector = load_and_preprocess_image(paths[2])
        if img_vector is not None:
            print(f"Shape đầu ra: {img_vector.shape}")
    else:
        print("Không tìm thấy ảnh nào")