import pickle
import numpy as np
import os  # Thêm import os

from sklearn.svm import SVC

def save_encodings(encodings, names, file_path):
    """
    Lưu encodings và tên.
    """
    with open(file_path, "wb") as f:
        pickle.dump((encodings, names), f)
    print(f"✅ Đã lưu encodings tại {file_path}")

def load_encodings(file_path):
    """
    Tải encodings và tên.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"❌ Không tìm thấy {file_path}")

def train_svm(X_train, y_train, model_path):
    """
    Huấn luyện và lưu SVM.
    """
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"✅ Mô hình SVM đã được lưu tại {model_path}")
    return svm