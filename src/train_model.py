import pickle
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Import hàm load_config và get_config_value từ config_manager.py
from src.config_manager import load_config, get_config_value
# Import ModelManager để lưu mô hình
from src.model_manager import ModelManager
# Import logger từ log_utils.py
from utils.log_utils import logger

# Đọc cấu hình
config = load_config()
encodings_file = get_config_value(config, 'paths', 'encodings_file', "../models/face_encodings.pkl")

def train_model():
    """
    Huấn luyện mô hình SVM với embeddings 512D từ file encodings.

    Quy trình:
    1. Tải dữ liệu encodings và nhãn.
    2. Kiểm tra số lượng lớp tối thiểu.
    3. Chia dữ liệu thành tập train/test.
    4. Huấn luyện SVM.
    5. Lưu mô hình và đánh giá.
    """
    # Tải dữ liệu encodings từ file
    if not os.path.exists(encodings_file):
        logger.error(f"File encodings không tồn tại: {encodings_file}")
        return

    with open(encodings_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    if len(known_face_encodings) == 0 or len(known_face_names) == 0:
        logger.error("Dữ liệu encodings hoặc nhãn trống.")
        return

    # Chuyển đổi dữ liệu thành mảng numpy
    X = np.array(known_face_encodings)  # Embeddings 512D
    y = np.array(known_face_names)      # Nhãn (tên người)

    # Mã hóa nhãn thành số nguyên
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Kiểm tra số lượng lớp tối thiểu
    unique_classes = len(label_encoder.classes_)
    if unique_classes < 2:
        logger.error("Cần ít nhất 2 lớp (người) để huấn luyện SVM.")
        return

    # Chia dữ liệu thành tập train (80%) và test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Huấn luyện mô hình SVM
    logger.info("Bắt đầu huấn luyện mô hình SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, probability=True)  # Sử dụng kernel RBF, C=1.0
    svm_model.fit(X_train, y_train)
    logger.info("Huấn luyện SVM hoàn tất!")

    # Lưu mô hình bằng ModelManager
    manager = ModelManager()
    manager.save_model(svm_model)

    # Đánh giá mô hình trên tập test
    logger.info("Đánh giá mô hình trên tập kiểm tra...")
    y_pred = svm_model.predict(X_test)
    print("\n🔍 Báo cáo phân loại:")  # Giữ print để in báo cáo trực tiếp
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # In số lượng mẫu và lớp
    logger.info(f"Tổng số mẫu: {len(X)}")
    logger.info(f"Số lớp (người): {len(label_encoder.classes_)}")

if __name__ == "__main__":
    # Chạy huấn luyện mô hình
    train_model()
    logger.info("Quá trình huấn luyện hoàn tất!")