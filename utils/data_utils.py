import os
import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(112, 112)):
    """
    Đọc và tiền xử lý ảnh cho InsightFace.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image_rgb, target_size)

def preprocess_batch(images_dir, target_size=(112, 112)):
    """
    Tiền xử lý hàng loạt ảnh.
    """
    images, labels = [], []
    for person_name in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    img = load_and_preprocess_image(image_path, target_size)
                    images.append(img)
                    labels.append(person_name)
                except Exception as e:
                    print(f"❌ Lỗi xử lý {image_path}: {e}")
    return np.array(images), np.array(labels)   