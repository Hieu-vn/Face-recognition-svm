import os
import argparse
from src.config_manager import load_config, get_config_value
from src.preprocess import preprocess_all_faces
from src.encode_faces import encode_faces
from src.train_model import train_model
from src.recognize import recognize
# Import logger từ log_utils.py
from utils.log_utils import logger

def run_pipeline(step=None, save_video=False, input_source=0):
    """
    Chạy pipeline nhận diện khuôn mặt:
    1. Xử lý ảnh khuôn mặt.
    2. Trích xuất embeddings.
    3. Huấn luyện mô hình SVM.
    4. Nhận diện thời gian thực.

    Args:
        step (str): Bước cụ thể để chạy (preprocess, encode, train, recognize). Nếu None, chạy toàn bộ pipeline.
        save_video (bool): Lưu video nhận diện nếu True.
        input_source (int/str): 0 cho webcam, hoặc đường dẫn đến file video/ảnh.
    """
    config = load_config()
    input_dir = get_config_value(config, 'data', 'known_faces_processed', "../data/known_faces_processed/")

    logger.info("Starting face recognition pipeline...")

    if step is None or step == "preprocess":
        logger.info("Step 1: Processing face images...")
        preprocess_all_faces()
        if not os.path.exists(input_dir):
            logger.error("Thư mục ảnh đã xử lý không tồn tại sau khi xử lý.")
            return

    if step is None or step == "encode":
        logger.info("Bước 2: Trích xuất embeddings...")
        encode_faces(input_dir)

    if step is None or step == "train":
        logger.info("Bước 3: Huấn luyện mô hình SVM...")
        train_model()

    if step is None or step == "recognize":
        logger.info("Bước 4: Bắt đầu nhận diện thời gian thực...")
        recognize(input_source=input_source, save_video=save_video)

    logger.info("Pipeline completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline nhận diện khuôn mặt.")
    parser.add_argument('--step', type=str, choices=['preprocess', 'encode', 'train', 'recognize'],
                        help="Chạy bước cụ thể (nếu không chọn, chạy toàn bộ pipeline).")
    parser.add_argument('--save-video', action='store_true', help="Lưu video nhận diện vào file.")
    parser.add_argument('--input-source', type=str, default=0, help="Nguồn đầu vào (0 cho webcam, hoặc đường dẫn file video/ảnh).")
    args = parser.parse_args()

    try:
        run_pipeline(step=args.step, save_video=args.save_video, input_source=args.input_source)
    except Exception as e:
        logger.error(f"Lỗi trong pipeline: {e}")