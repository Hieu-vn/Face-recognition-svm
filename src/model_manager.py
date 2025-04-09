import os
import pickle

# Import hàm load_config và get_config_value từ config_manager.py (nếu cần)
from src.config_manager import load_config, get_config_value
# Import logger từ log_utils.py
from utils.log_utils import logger

class ModelManager:
    """
    Quản lý việc tải và lưu mô hình SVM trong dự án nhận diện khuôn mặt.
    """
    def __init__(self, model_type="svm", model_path=None):
        """
        Khởi tạo ModelManager.

        Args:
            model_type (str): Loại mô hình (mặc định là "svm").
            model_path (str): Đường dẫn đến file mô hình, nếu None sẽ lấy từ config.yaml.
        """
        self.model_type = model_type
        self.config = load_config() if model_path is None else None
        self.model_path = model_path if model_path else get_config_value(self.config, 'paths', 'svm_model_file', "../models/face_recognition_svm.pkl")
        self.model = None
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def load_model(self):
        """
        Tải mô hình từ file.

        Returns:
            object: Mô hình SVM nếu tải thành công, None nếu thất bại.

        Raises:
            FileNotFoundError: Nếu file mô hình không tồn tại.
        """
        if self.model_type != "svm":
            logger.error(f"Loại mô hình '{self.model_type}' không được hỗ trợ.")
            return None

        if not os.path.exists(self.model_path):
            logger.error(f"File mô hình không tồn tại: {self.model_path}")
            return None

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Mô hình SVM đã được tải từ: {self.model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")
            return None

    def save_model(self, model):
        """
        Lưu mô hình vào file.

        Args:
            model: Mô hình SVM cần lưu.

        Returns:
            bool: True nếu lưu thành công, False nếu thất bại.
        """
        if self.model_type != "svm":
            logger.error(f"Loại mô hình '{self.model_type}' không được hỗ trợ.")
            return False

        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Mô hình SVM đã được lưu tại: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")
            return False

    def check_model_exists(self):
        """
        Kiểm tra xem file mô hình có tồn tại không.

        Returns:
            bool: True nếu tồn tại, False nếu không.
        """
        return os.path.exists(self.model_path)

if __name__ == "__main__":
    # Ví dụ sử dụng
    # Khởi tạo ModelManager
    manager = ModelManager()

    # Tải mô hình (giả định đã có file)
    model = manager.load_model()
    if model:
        logger.info("Mô hình đã tải thành công!")

    # Lưu mô hình (giả định có mô hình để lưu)
    from sklearn.svm import SVC
    dummy_model = SVC()  # Mô hình giả lập
    if manager.save_model(dummy_model):
        logger.info("Mô hình đã lưu thành công!")

    # Kiểm tra tồn tại
    logger.info(f"File mô hình tồn tại: {manager.check_model_exists()}")