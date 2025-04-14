import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
import time

# Import các module cần thiết
from src.config_manager import load_config, get_config_value
from src.path_manager import path_manager
from src.model_manager import model_manager
from utils.logger_init import logger
from src.image_processor import ImageProcessor

# Đọc cấu hình
config = load_config()
distance_threshold = get_config_value(config, 'insightface', 'distance_threshold', 0.6)
process_frame_interval = get_config_value(config, 'recognition', 'process_frame_interval', 10)
output_video_path = path_manager.get_path('output_video')

# Khởi tạo InsightFace
model_name = get_config_value(config, 'insightface', 'model_name', 'buffalo_l')
det_size_list = get_config_value(config, 'insightface', 'det_size', [320, 320])
det_size = tuple(det_size_list)

app = FaceAnalysis(
    name=model_name,
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=-1, det_size=det_size)

def process_frame(frame, svm_model, label_encoder, known_face_encodings):
    """
    Xử lý một khung hình để phát hiện và nhận diện khuôn mặt.
    
    Args:
        frame: Khung hình cần xử lý.
        svm_model: Mô hình SVM đã huấn luyện.
        label_encoder: LabelEncoder đã fit với tên người.
        known_face_encodings: Danh sách embeddings đã biết.
        
    Returns:
        tuple: (face_locations, face_names) - Vị trí và tên của các khuôn mặt.
    """
    scale_factor = 0.25
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    faces = app.get(small_frame)
    if not faces:
        logger.info("Không phát hiện khuôn mặt trong khung hình.")
    else:
        logger.info(f"Phát hiện {len(faces)} khuôn mặt trong khung hình.")

    face_locations = [(int(f.bbox[1]/scale_factor), int(f.bbox[2]/scale_factor), int(f.bbox[3]/scale_factor), int(f.bbox[0]/scale_factor)) for f in faces]
    face_encodings = [f.normed_embedding for f in faces]

    face_names = []
    for face_encoding in face_encodings:
        svm_prediction = svm_model.predict([face_encoding])[0]
        svm_name = label_encoder.inverse_transform([svm_prediction])[0]
        distances = [cosine(face_encoding, known_encoding) for known_encoding in known_face_encodings]
        min_distance = min(distances) if distances else float('inf')
        name = svm_name if min_distance < distance_threshold else "Unknown"
        logger.info(f"Dự đoán: {name} (khoảng cách nhỏ nhất: {min_distance:.4f})")
        face_names.append(name)

    return face_locations, face_names

def recognize(input_source=0, save_video=False, output_video_path=output_video_path):
    """
    Nhận diện khuôn mặt từ nguồn đầu vào (webcam, video hoặc ảnh).
    
    Args:
        input_source: 0 cho webcam, hoặc đường dẫn đến file video/ảnh.
        save_video: True nếu muốn lưu video nhận diện.
        output_video_path: Đường dẫn lưu video nhận diện.
    """
    # Tải encodings và mô hình
    known_face_encodings, known_face_names = model_manager.load_encodings()
    if known_face_encodings is None or known_face_names is None:
        logger.error("Không thể tải dữ liệu encodings.")
        return

    svm_model = model_manager.load_model()
    if svm_model is None:
        logger.error("Không thể tải mô hình SVM.")
        return

    label_encoder = LabelEncoder()
    label_encoder.fit(known_face_names)

    # Xác định loại nguồn đầu vào
    if isinstance(input_source, int) and input_source == 0:
        cap = cv2.VideoCapture(0)
        source_type = "webcam"
    else:
        cap = cv2.VideoCapture(input_source)
        source_type = "video" if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1 else "image"
        if not cap.isOpened():
            logger.error(f"Không thể mở file: {input_source}")
            return

    if not cap.isOpened():
        logger.error("Không thể mở nguồn đầu vào.")
        return

    logger.info(f"Bắt đầu nhận diện từ {source_type}...")
    frame_count = 0
    last_face_locations = []
    last_face_names = []
    last_update_time = 0
    display_duration = 2  # Thời gian giữ tên trên màn hình (giây)

    # Thiết lập video writer nếu cần
    video_writer = None
    if save_video and source_type != "image":
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
        logger.info(f"Đang ghi video vào: {output_video_path}")

    # Vòng lặp chính
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Kết thúc nhận diện từ {source_type}.")
            break

        frame_count += 1
        current_time = time.time()

        # Xử lý frame theo interval
        if frame_count % process_frame_interval == 0:
            face_locations, face_names = process_frame(frame, svm_model, label_encoder, known_face_encodings)
            if face_locations:  # Chỉ cập nhật nếu phát hiện khuôn mặt
                last_face_locations = face_locations
                last_face_names = face_names
                last_update_time = current_time

        # Hiển thị tên nếu vẫn trong khoảng thời gian display_duration
        if current_time - last_update_time <= display_duration:
            for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Recognition", frame)
        if video_writer:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Nhận diện kết thúc.")

def recognize_face(image):
    """Nhận diện khuôn mặt trong ảnh"""
    try:
        # Kiểm tra model
        if not hasattr(model_manager, 'model') or model_manager.model is None:
            logger.error("Model chưa được huấn luyện")
            return None, 0.0
            
        # Khởi tạo ImageProcessor
        image_processor = ImageProcessor()
        
        # Phát hiện và cắt khuôn mặt
        face_img, face_info = image_processor.detect_and_crop_face(image)
        if face_img is None:
            return None, 0.0
            
        # Lấy embedding
        embedding = model_manager.get_embedding(face_img)
        if embedding is None:
            return None, 0.0
            
        # Dự đoán
        # Kiểm tra nếu model_manager có phương thức predict
        if hasattr(model_manager, 'predict'):
            label, prob = model_manager.predict(embedding)
        else:
            # Nếu không có, tự thực hiện dự đoán
            predicted_label = model_manager.model.predict([embedding])[0]
            prob = max(model_manager.model.predict_proba([embedding])[0])
            # Lấy nhãn tương ứng
            encodings, names = model_manager.load_encodings()
            label = predicted_label
        
        # Kiểm tra ngưỡng tin cậy
        confidence_threshold = 0.6
        if hasattr(model_manager, 'config'):
            confidence_threshold = get_config_value(model_manager.config, 'recognition', 'min_confidence', 0.6)
        
        if prob < confidence_threshold:
            logger.warning(f"Độ tin cậy thấp ({prob:.2f} < {confidence_threshold})")
            return None, prob
            
        return label, prob
        
    except Exception as e:
        logger.error(f"Lỗi khi nhận diện khuôn mặt: {e}")
        return None, 0.0

def recognize_faces_in_video(video_path=None):
    """Nhận diện khuôn mặt trong video"""
    try:
        # Mở camera hoặc video
        if video_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
            
        if not cap.isOpened():
            logger.error("Không thể mở camera hoặc video")
            return False
            
        # Lấy thông tin video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Tạo video writer
        output_path = path_manager.get_path('output_video')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Khởi tạo ImageProcessor
        image_processor = ImageProcessor()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Phát hiện và cắt khuôn mặt
            face_img, face_info = image_processor.detect_and_crop_face(frame)
            if face_img is not None:
                # Lấy embedding
                embedding = model_manager.get_embedding(face_img)
                if embedding is not None:
                    # Dự đoán
                    label, prob = model_manager.predict(embedding)
                    
                    # Vẽ kết quả
                    if label is not None:
                        x1, y1, x2, y2 = map(int, face_info.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({prob:.2f})", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
            # Ghi frame
            out.write(frame)
            
            # Hiển thị frame
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Giải phóng tài nguyên
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Đã lưu video kết quả vào {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi nhận diện khuôn mặt trong video: {e}")
        return False

if __name__ == "__main__":
    recognize_faces_in_video()
