import os
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

# Import các module cần thiết
from src.config_manager import load_config, get_config_value
from src.path_manager import path_manager
from src.model_manager import model_manager
from utils.logger_init import logger

def train_model(encodings_file=None, model_file=None, params_search=False):
    """
    Huấn luyện mô hình SVM từ file encodings và lưu mô hình.
    
    Args:
        encodings_file (str, optional): Đường dẫn file embeddings.
        model_file (str, optional): Đường dẫn file mô hình.
        params_search (bool): Thực hiện tìm kiếm tham số tối ưu (chậm hơn).
    
    Returns:
        bool: True nếu huấn luyện thành công, False nếu thất bại.
    """
    try:
        # Sử dụng đường dẫn mặc định nếu không có đường dẫn truyền vào
        if encodings_file is None:
            encodings_file = path_manager.get_path('encodings_file')
        if model_file is None:
            model_file = path_manager.get_path('svm_model_file')
            
        # Kiểm tra thư mục đầu vào
        if not os.path.exists(encodings_file):
            logger.error(f"File embeddings không tồn tại: {encodings_file}")
            return False
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        
        # Tải embeddings
        logger.info("Tải embeddings từ file...")
        embeddings, names = model_manager.load_encodings()
        
        if embeddings is None or names is None:
            logger.error("Không thể tải embeddings")
            return False
            
        if len(embeddings) == 0 or len(names) == 0:
            logger.error("Không có dữ liệu embeddings")
            return False
            
        logger.info(f"Đã tải {len(embeddings)} embeddings cho {len(set(names))} người")
        
        # Chuyển đổi sang numpy array
        X = np.array(embeddings)
        y = np.array(names)
        
        # Mã hóa nhãn thành số nguyên
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Kiểm tra số lượng lớp tối thiểu
        unique_classes = len(label_encoder.classes_)
        if unique_classes < 2:
            logger.error("Cần ít nhất 2 lớp (người) để huấn luyện SVM")
            return False
            
        # Thống kê số lượng mẫu cho mỗi lớp
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Phân bố dữ liệu:")
        for i, (name, count) in enumerate(zip(unique, counts)):
            logger.info(f"  {name}: {count} mẫu")
        
        # Chia dữ liệu thành tập train và test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
        logger.info(f"Tập kiểm tra: {X_test.shape[0]} mẫu")
        
        # Huấn luyện mô hình SVM
        logger.info("Bắt đầu huấn luyện mô hình SVM...")
        start_time = time.time()
        
        if params_search:
            # Tìm kiếm siêu tham số tối ưu
            logger.info("Đang tìm kiếm siêu tham số tối ưu (có thể mất nhiều thời gian)...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['linear', 'rbf']
            }
            grid_search = GridSearchCV(
                SVC(probability=True), 
                param_grid, 
                cv=5,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            # Lấy mô hình tốt nhất
            best_params = grid_search.best_params_
            logger.info(f"Tham số tối ưu: {best_params}")
            svm_model = grid_search.best_estimator_
        else:
            # Sử dụng tham số mặc định
            svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
            svm_model.fit(X_train, y_train)
        
        # Đánh giá mô hình
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Tính thời gian huấn luyện
        training_time = time.time() - start_time
        logger.info(f"Huấn luyện hoàn tất trong {training_time:.2f} giây")
        logger.info(f"Độ chính xác trên tập kiểm tra: {accuracy:.4f}")
        
        # In báo cáo phân loại chi tiết
        class_names = label_encoder.classes_
        classification_rep = classification_report(y_test, y_pred, target_names=class_names)
        logger.info(f"Báo cáo phân loại:\n{classification_rep}")
        
        # Tạo thư mục cho kết quả
        results_dir = os.path.join(os.path.dirname(model_file), 'evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        # Lưu báo cáo vào file
        with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Độ chính xác: {accuracy:.4f}\n")
            f.write(f"Thời gian huấn luyện: {training_time:.2f} giây\n")
            f.write("\nBáo cáo phân loại:\n")
            f.write(classification_rep)
        
        # Lưu mô hình
        if model_manager.save_model(svm_model):
            logger.info(f"Đã lưu mô hình vào {model_file}")
            
            # Lưu thông tin lớp
            with open(os.path.join(results_dir, 'class_info.txt'), 'w') as f:
                f.write("Thông tin các lớp:\n")
                for i, name in enumerate(class_names):
                    f.write(f"{i}: {name}\n")
            
            return True
        else:
            logger.error("Không thể lưu mô hình")
            return False
        
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_model(test_img_path=None):
    """
    Kiểm tra mô hình với một ảnh cụ thể
    
    Args:
        test_img_path (str): Đường dẫn đến ảnh kiểm tra.
        
    Returns:
        tuple: (predicted_name, confidence) hoặc (None, 0) nếu thất bại.
    """
    try:
        import cv2
        from src.image_processor import ImageProcessor
        
        # Kiểm tra đường dẫn
        if test_img_path is None or not os.path.exists(test_img_path):
            logger.error(f"Ảnh kiểm tra không tồn tại: {test_img_path}")
            return None, 0
            
        # Tải mô hình
        model = model_manager.load_model()
        if model is None:
            logger.error("Không thể tải mô hình")
            return None, 0
            
        # Tải embeddings để lấy tên
        encodings, names = model_manager.load_encodings()
        if encodings is None or names is None:
            logger.error("Không thể tải dữ liệu embeddings")
            return None, 0
            
        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        label_encoder.fit(names)
        
        # Tạo image processor
        image_processor = ImageProcessor()
        
        # Đọc ảnh
        image = cv2.imread(test_img_path)
        if image is None:
            logger.error(f"Không thể đọc ảnh: {test_img_path}")
            return None, 0
            
        # Phát hiện và cắt khuôn mặt
        face_img, _ = image_processor.detect_and_crop_face(image, test_img_path)
        if face_img is None:
            logger.error("Không phát hiện được khuôn mặt trong ảnh")
            return None, 0
            
        # Lấy embedding
        embedding = model_manager.get_embedding(face_img)
        if embedding is None:
            logger.error("Không thể trích xuất embedding")
            return None, 0
            
        # Dự đoán
        predicted_label = model.predict([embedding])[0]
        probs = model.predict_proba([embedding])[0]
        confidence = probs[predicted_label]
        
        # Chuyển đổi nhãn thành tên
        predicted_name = label_encoder.inverse_transform([predicted_label])[0]
        
        logger.info(f"Dự đoán: {predicted_name} (tin cậy: {confidence:.4f})")
        return predicted_name, confidence
        
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra mô hình: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None, 0

if __name__ == "__main__":
    train_model(params_search=False)