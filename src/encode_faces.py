import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Import hàm load_config và get_config_value từ config_manager.py
from src.config_manager import load_config, get_config_value
# Import logger từ log_utils.py
from utils.log_utils import logger

# Đọc cấu hình
config = load_config()
input_dir = get_config_value(config, 'data', 'known_faces_processed', "../data/known_faces_processed/")
encodings_file = get_config_value(config, 'paths', 'encodings_file', "../models/face_encodings.pkl")
os.makedirs(os.path.dirname(encodings_file), exist_ok=True)

# Khởi tạo InsightFace với ArcFace cho trích xuất embeddings
app = FaceAnalysis(
    name=get_config_value(config, 'insightface', 'model_name', 'buffalo_l'),
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=tuple(get_config_value(config, 'insightface', 'det_size', [640, 640])))

def encode_faces(image_dir):
    """
    Trích xuất embeddings 512D từ ảnh khuôn mặt bằng ArcFace và lưu vào file encodings.

    Args:
        image_dir (str): Đường dẫn đến thư mục chứa ảnh đã xử lý.
    """
    known_face_encodings = []  # Danh sách embeddings 512D
    known_face_names = []      # Danh sách tên người tương ứng

    # Duyệt qua từng thư mục con (mỗi thư mục là một người)
    for person_name in os.listdir(image_dir):
        person_dir = os.path.join(image_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        # Duyệt qua từng ảnh trong thư mục
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            if os.path.isfile(image_path) and image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Đọc ảnh
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Không thể đọc ảnh: {image_path}")
                    continue

                # Trích xuất embeddings bằng ArcFace
                faces = app.get(image)
                if len(faces) == 0:
                    logger.error(f"Không tìm thấy khuôn mặt trong ảnh: {image_path}")
                    continue
                if len(faces) > 1:
                    logger.warning(f"Tìm thấy nhiều khuôn mặt trong ảnh: {image_path}, chỉ lấy khuôn mặt đầu tiên")

                face = faces[0]
                embedding = face.normed_embedding  # Lấy embedding 512D đã chuẩn hóa
                known_face_encodings.append(embedding)
                known_face_names.append(person_name)
                logger.info(f"Đã trích xuất embedding cho {person_name} từ ảnh: {image_name}")

    # Kiểm tra số lượng embeddings trước khi lưu
    if len(known_face_encodings) < 1:
        logger.error("Không có embeddings nào được trích xuất, kiểm tra lại dữ liệu ảnh.")
        return

    # Lưu encodings và tên vào file
    with open(encodings_file, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    logger.info(f"Đã lưu encodings vào tệp: {encodings_file}")

if __name__ == "__main__":
    # Chạy trích xuất embeddings
    encode_faces(input_dir)
    logger.info("Trích xuất embeddings hoàn tất!")