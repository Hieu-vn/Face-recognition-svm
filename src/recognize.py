import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
from src.config_manager import load_config, get_config_value
from src.model_manager import ModelManager
from utils.log_utils import logger
import time

config = load_config()
encodings_file = get_config_value(config, 'paths', 'encodings_file', "models/face_encodings.pkl")
svm_model_file = get_config_value(config, 'paths', 'svm_model_file', "models/face_recognition_svm.pkl")
distance_threshold = get_config_value(config, 'insightface', 'distance_threshold', 0.6)
process_frame_interval = get_config_value(config, 'recognition', 'process_frame_interval', 10)
output_video_path = get_config_value(config, 'paths', 'output_video_path', "output/recognition_output.mp4")

app = FaceAnalysis(
    name=get_config_value(config, 'insightface', 'model_name', 'buffalo_l'),
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=-1, det_size=tuple(get_config_value(config, 'insightface', 'det_size', [320, 320])))

def process_frame(frame, svm_model, label_encoder, known_face_encodings):
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
    if not os.path.exists(encodings_file):
        logger.error(f"File encodings không tồn tại: {encodings_file}")
        return

    with open(encodings_file, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    manager = ModelManager()
    svm_model = manager.load_model()
    if svm_model is None:
        logger.error("Không thể tải mô hình SVM.")
        return

    label_encoder = LabelEncoder()
    label_encoder.fit(known_face_names)

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

    video_writer = None
    if save_video and source_type != "image":
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
        logger.info(f"Đang ghi video vào: {output_video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Kết thúc nhận diện từ {source_type}.")
            break

        frame_count += 1
        current_time = time.time()

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

    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Nhận diện kết thúc.")

if __name__ == "__main__":
    recognize()