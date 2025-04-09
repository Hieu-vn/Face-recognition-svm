import os
import cv2
from insightface.app import FaceAnalysis
from src.config_manager import load_config, get_config_value
from utils.log_utils import logger

def preprocess_image(image_path, output_dir):
    config = load_config()
    app = FaceAnalysis(
        name=get_config_value(config, 'insightface', 'model_name', 'buffalo_l'),
        allowed_modules=['detection']
    )
    app.prepare(ctx_id=-1, det_size=tuple(get_config_value(config, 'insightface', 'det_size', [320, 320])))

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Không thể đọc ảnh: {image_path}")
        return False

    faces = app.get(img)
    if not faces:
        logger.warning(f"Không phát hiện khuôn mặt trong ảnh: {image_path}")
        return False

    face = faces[0]  # Lấy khuôn mặt đầu tiên
    bbox = face.bbox.astype(int)
    x, y, x2, y2 = bbox
    x, y, x2, y2 = max(0, x), max(0, y), min(img.shape[1], x2), min(img.shape[0], y2)
    face_img = img[y:y2, x:x2]
    face_img = cv2.resize(face_img, (112, 112))

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, face_img)
    logger.info(f"Processed and saved image: {output_path}")
    return True

def preprocess_all_faces():
    config = load_config()
    input_dir = get_config_value(config, 'data', 'known_faces', "../data/known_faces/")
    output_dir_base = get_config_value(config, 'data', 'known_faces_processed', "../data/known_faces_processed/")

    logger.info("Bắt đầu xử lý ảnh khuôn mặt...")
    total_processed = 0
    for person_folder in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_folder)
        if os.path.isdir(person_path):
            output_dir = os.path.join(output_dir_base, person_folder)
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, filename)
                    if preprocess_image(image_path, output_dir):
                        total_processed += 1

    logger.info(f"Hoàn tất xử lý {total_processed} ảnh khuôn mặt!")

if __name__ == "__main__":
    preprocess_all_faces()