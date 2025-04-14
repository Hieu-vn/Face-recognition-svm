import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from insightface.app import FaceAnalysis
import cv2
import time

# Import các module cần thiết
from src.config_manager import load_config, get_config_value
from src.path_manager import path_manager
from utils.logger_init import logger

class ModelManager:
    """
    Quản lý việc tải, lưu và huấn luyện mô hình trong dự án nhận diện khuôn mặt.
    """
    def __init__(self, model_type="svm", model_path=None):
        """
        Khởi tạo ModelManager.

        Args:
            model_type (str): Loại mô hình (mặc định là "svm").
            model_path (str): Đường dẫn đến file mô hình, nếu None sẽ lấy từ config.yaml.
        """
        self.model_type = model_type
        self.config = load_config()
        
        # Nếu không cung cấp model_path, lấy từ path_manager
        self.model_path = model_path if model_path else path_manager.get_path('svm_model_file')
        self.encodings_path = path_manager.get_path('encodings_file')
        self.model = None
        
        # Đảm bảo thư mục cho model_path và encodings_path tồn tại
        os.makedirs(os.path.dirname(str(self.model_path)), exist_ok=True)
        os.makedirs(os.path.dirname(str(self.encodings_path)), exist_ok=True)
        
        # Khởi tạo InsightFace model
        self.face_app = None
        
    def _initialize_face_app(self):
        """
        Khởi tạo model InsightFace nếu chưa được khởi tạo.
        """
        if self.face_app is None:
            try:
                model_name = get_config_value(self.config, 'insightface', 'model_name', 'buffalo_l')
                
                # Lấy det_size từ config và chuyển đổi thành tuple
                det_size_list = get_config_value(self.config, 'insightface', 'det_size', [640, 640])
                det_size = tuple(det_size_list)
                
                logger.info(f"Khởi tạo InsightFace với model {model_name}")
                self.face_app = FaceAnalysis(
                    name=model_name,
                    providers=['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=-1, det_size=det_size)
                logger.info("InsightFace đã được khởi tạo thành công")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo InsightFace: {e}")
                self.face_app = None
                
    def get_embedding(self, face_img):
        """
        Trích xuất embedding từ ảnh khuôn mặt.
        
        Args:
            face_img: Ảnh khuôn mặt đã được cắt và xử lý.
            
        Returns:
            numpy.ndarray: Embedding 512D, hoặc None nếu thất bại.
        """
        try:
            # Khởi tạo InsightFace nếu cần
            if self.face_app is None:
                self._initialize_face_app()
                if self.face_app is None:
                    logger.error("InsightFace model chưa được khởi tạo")
                    return None
            
            # Kiểm tra ảnh đầu vào
            if face_img is None or not isinstance(face_img, np.ndarray):
                logger.error("Ảnh đầu vào không hợp lệ hoặc không phải là numpy array")
                return None
                
            # Đảm bảo ảnh có 3 kênh màu
            if len(face_img.shape) == 2:  # grayscale
                logger.debug("Chuyển đổi ảnh từ grayscale sang BGR")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            elif face_img.shape[2] == 4:  # RGBA
                logger.debug("Chuyển đổi ảnh từ RGBA sang BGR")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
            
            # Kiểm tra kích thước ảnh - InsightFace yêu cầu ảnh đủ lớn
            h, w = face_img.shape[:2]
            min_face_size = 80  # Kích thước tối thiểu cho phép phát hiện khuôn mặt
            
            if min(h, w) < min_face_size:
                logger.warning(f"Ảnh khuôn mặt quá nhỏ ({w}x{h}), đang resize")
                scale_factor = min_face_size / min(h, w)
                face_img = cv2.resize(face_img, None, fx=scale_factor, fy=scale_factor)
            
            # Tạo nhiều phiên bản của ảnh với các cải tiến khác nhau để tăng khả năng phát hiện
            enhanced_versions = []
            
            # 1. Ảnh gốc
            enhanced_versions.append(face_img.copy())
            
            # 2. Tăng độ tương phản
            enhanced = cv2.convertScaleAbs(face_img, alpha=1.5, beta=10)
            enhanced_versions.append(enhanced)
            
            # 3. Tăng độ sáng
            bright = cv2.convertScaleAbs(face_img, alpha=1.0, beta=50)
            enhanced_versions.append(bright)
            
            # 4. Cân bằng histogram
            if len(face_img.shape) == 3:
                lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                enhanced_hist = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                enhanced_versions.append(enhanced_hist)
            
            # 5. Thêm scale lớn hơn
            larger_scale = cv2.resize(face_img, None, fx=1.5, fy=1.5)
            enhanced_versions.append(larger_scale)
            
            # Thử phát hiện khuôn mặt với mỗi phiên bản cải tiến
            for version in enhanced_versions:
                faces = self.face_app.get(version)
                if len(faces) > 0 and hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                    embedding = faces[0].embedding
                    return embedding
            
            # Nếu vẫn không phát hiện được, ghi log và lưu ảnh để debug
            logger.warning("Không phát hiện khuôn mặt nào trong ảnh đầu vào sau nhiều phương pháp")
            try:
                debug_dir = os.path.join(os.path.dirname(str(self.encodings_path)), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                debug_file = os.path.join(debug_dir, f'embedding_error_{int(time.time())}.jpg')
                cv2.imwrite(debug_file, face_img)
                logger.debug(f"Đã lưu ảnh lỗi embedding tại {debug_file}")
            except Exception as debug_err:
                logger.debug(f"Không thể lưu ảnh debug: {debug_err}")
            return None
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất embedding: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def load_model(self):
        """
        Tải mô hình từ file.

        Returns:
            object: Mô hình SVM nếu tải thành công, None nếu thất bại.
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
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(str(self.model_path)), exist_ok=True)
            
            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)
            
            self.model = model  # Lưu model vào biến thành viên
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

    def save_encodings(self, encodings, names):
        """
        Lưu encodings và tên vào file.

        Args:
            encodings: Danh sách embeddings 512D.
            names: Danh sách tên người tương ứng.

        Returns:
            bool: True nếu lưu thành công, False nếu thất bại.
        """
        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(str(self.encodings_path)), exist_ok=True)
            
            with open(self.encodings_path, "wb") as f:
                pickle.dump((encodings, names), f)
            logger.info(f"Đã lưu encodings tại {self.encodings_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu encodings: {e}")
            return False

    def load_encodings(self):
        """
        Tải encodings và tên từ file.

        Returns:
            tuple: (encodings, names) nếu tải thành công, (None, None) nếu thất bại.
        """
        if not os.path.exists(self.encodings_path):
            logger.error(f"File encodings không tồn tại: {self.encodings_path}")
            return None, None

        try:
            with open(self.encodings_path, "rb") as f:
                encodings, names = pickle.load(f)
            logger.info(f"Đã tải encodings từ {self.encodings_path}")
            return encodings, names
        except Exception as e:
            logger.error(f"Lỗi khi tải encodings: {e}")
            return None, None

    def train_svm(self, X_train=None, y_train=None):
        """
        Huấn luyện mô hình SVM.

        Args:
            X_train: Dữ liệu huấn luyện (nếu None, sẽ tải từ file encodings).
            y_train: Nhãn huấn luyện (nếu None, sẽ tải từ file encodings).

        Returns:
            object: Mô hình SVM đã huấn luyện nếu thành công, None nếu thất bại.
        """
        # Nếu không có dữ liệu huấn luyện, tải từ file encodings
        if X_train is None or y_train is None:
            encodings, names = self.load_encodings()
            if encodings is None or names is None:
                logger.error("Không thể tải encodings để huấn luyện mô hình")
                return None
            
            X_train = np.array(encodings)
            y_train = np.array(names)
        
        # Kiểm tra số lượng mẫu
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("Không có dữ liệu huấn luyện")
            return None
            
        # Kiểm tra kích thước dữ liệu
        if len(X_train) != len(y_train):
            logger.error(f"Số lượng mẫu dữ liệu ({len(X_train)}) và nhãn ({len(y_train)}) không khớp")
            return None

        # Mã hóa nhãn thành số nguyên
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train)

        # Kiểm tra số lượng lớp tối thiểu
        unique_classes = len(label_encoder.classes_)
        if unique_classes < 2:
            logger.error("Cần ít nhất 2 lớp (người) để huấn luyện SVM.")
            return None

        # Chia dữ liệu thành tập train (80%) và test (20%)
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_encoded, test_size=0.2, random_state=42
        )

        # Huấn luyện mô hình SVM
        try:
            logger.info("Bắt đầu huấn luyện mô hình SVM...")
            # Lấy tham số SVM từ config nếu có
            kernel = get_config_value(self.config, "svm.kernel", "rbf")
            C = get_config_value(self.config, "svm.C", 1.0)
            gamma = get_config_value(self.config, "svm.gamma", "scale")
            probability = get_config_value(self.config, "svm.probability", True)
            
            svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability)
            svm_model.fit(X_train_split, y_train_split)
            logger.info("Huấn luyện SVM hoàn tất!")
            
            # Lưu mô hình
            if self.save_model(svm_model):
                # Lưu label_encoder cùng với mô hình
                self.model = svm_model
                
                # Đánh giá mô hình trên tập test
                logger.info("Đánh giá mô hình trên tập kiểm tra...")
                y_pred = svm_model.predict(X_test)
                print("\n🔍 Báo cáo phân loại:")
                print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

                # In số lượng mẫu và lớp
                logger.info(f"Tổng số mẫu: {len(X_train)}")
                logger.info(f"Số lớp (người): {len(label_encoder.classes_)}")
                
                return svm_model
            else:
                logger.error("Không thể lưu mô hình đã huấn luyện")
                return None
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện SVM: {e}")
            return None

# Tạo instance mặc định
model_manager = ModelManager()

if __name__ == "__main__":
    # Ví dụ sử dụng
    # Khởi tạo ModelManager
    manager = ModelManager()

    # Tải mô hình (giả định đã có file)
    model = manager.load_model()
    if model:
        logger.info("Mô hình đã tải thành công!")

    # Lưu mô hình (giả định có mô hình để lưu)
    dummy_model = SVC()  # Mô hình giả lập
    if manager.save_model(dummy_model):
        logger.info("Mô hình đã lưu thành công!")

    # Kiểm tra tồn tại
    logger.info(f"File mô hình tồn tại: {manager.check_model_exists()}")