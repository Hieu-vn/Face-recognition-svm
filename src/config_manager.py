import os
import yaml

def load_config(config_file="../config/config.yaml"):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File cấu hình không tồn tại: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as file:  # Thêm encoding='utf-8'
        return yaml.safe_load(file)

def get_config_value(config, section, key, default=None):
    return config.get(section, {}).get(key, default)

def validate_config(config):
    """
    Kiểm tra tính hợp lệ của cấu hình.

    Args:
        config (dict): Dictionary cấu hình.

    Raises:
        ValueError: Nếu cấu hình không hợp lệ.
    """
    det_size = config.get('insightface', {}).get('det_size', [640, 640])
    if not isinstance(det_size, list) or len(det_size) != 2:
        raise ValueError("det_size phải là một list chứa 2 số nguyên.")
    if not all(isinstance(x, int) for x in det_size):
        raise ValueError("Các phần tử trong det_size phải là số nguyên.")
    distance_threshold = config.get('insightface', {}).get('distance_threshold', 0.6)
    if not isinstance(distance_threshold, (int, float)) or distance_threshold < 0 or distance_threshold > 1:
        raise ValueError("distance_threshold phải là số trong khoảng [0, 1].")
    return True


if __name__ == "__main__":
    # Ví dụ sử dụng
    try:
        config = load_config()
        print("Cấu hình đã tải:")
        print(f"Đường dẫn encodings_file: {get_config_value(config, 'paths', 'encodings_file')}")
        print(f"Mô hình InsightFace: {get_config_value(config, 'insightface', 'model_name')}")
        print(f"Ngưỡng khoảng cách: {get_config_value(config, 'insightface', 'distance_threshold')}")
    except Exception as e:
        print(f"❌ Lỗi khi chạy ví dụ: {e}")