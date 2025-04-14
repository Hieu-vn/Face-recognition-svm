import os
import yaml
from pathlib import Path
from utils.logger_init import logger

def load_config(config_path=None):
    """
    Load cấu hình từ file YAML.
    
    Args:
        config_path (str, optional): Đường dẫn đến file config. Nếu None, sử dụng đường dẫn mặc định.
        
    Returns:
        dict: Cấu hình đã load.
    """
    try:
        if config_path is None:
            # Sử dụng đường dẫn mặc định
            base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = base_path / 'config' / 'config.yaml'
            
        if not os.path.exists(config_path):
            logger.error(f"File cấu hình không tồn tại: {config_path}")
            return {}
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if not config:
            logger.warning("File cấu hình trống")
            return {}
            
        return config
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc file cấu hình: {e}")
        return {}

def get_config_value(config, section, key, default=None):
    """
    Lấy giá trị cấu hình từ section và key.
    
    Args:
        config (dict): Cấu hình đã load.
        section (str): Tên section.
        key (str): Tên key.
        default: Giá trị mặc định nếu không tìm thấy.
        
    Returns:
        Giá trị cấu hình hoặc giá trị mặc định.
    """
    try:
        return config.get(section, {}).get(key, default)
    except Exception as e:
        logger.error(f"Lỗi khi lấy giá trị cấu hình [{section}][{key}]: {e}")
        return default

def save_config(config, config_path=None):
    """
    Lưu cấu hình vào file YAML.
    
    Args:
        config (dict): Cấu hình cần lưu.
        config_path (str, optional): Đường dẫn đến file config. Nếu None, sử dụng đường dẫn mặc định.
        
    Returns:
        bool: True nếu thành công, False nếu thất bại.
    """
    try:
        if config_path is None:
            # Sử dụng đường dẫn mặc định
            base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = base_path / 'config' / 'config.yaml'
            
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"Đã lưu cấu hình vào: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu file cấu hình: {e}")
        return False

def validate_config(config):
    """
    Kiểm tra tính hợp lệ của các giá trị trong config
    
    Args:
        config (dict): Dictionary chứa cấu hình
    """
    # Kiểm tra cấu hình InsightFace
    insightface_config = config.get('insightface', {})
    if 'model_name' in insightface_config:
        valid_models = ['buffalo_l', 'buffalo_sc', 'antelopev2']
        if insightface_config['model_name'] not in valid_models:
            raise ValueError(f"Model name không hợp lệ. Phải là một trong: {valid_models}")
    
    if 'distance_threshold' in insightface_config:
        threshold = insightface_config['distance_threshold']
        if not 0 <= threshold <= 1:
            raise ValueError("distance_threshold phải nằm trong khoảng [0, 1]")
    
    # Kiểm tra cấu hình recognition
    recognition_config = config.get('recognition', {})
    if 'process_frame_interval' in recognition_config:
        interval = recognition_config['process_frame_interval']
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("process_frame_interval phải là số nguyên dương")
    
    if 'min_confidence' in recognition_config:
        confidence = recognition_config['min_confidence']
        if not 0 <= confidence <= 1:
            raise ValueError("min_confidence phải nằm trong khoảng [0, 1]")
    
    # Kiểm tra cấu hình logging
    logging_config = config.get('logging', {})
    if 'level' in logging_config:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if logging_config['level'] not in valid_levels:
            raise ValueError(f"Log level không hợp lệ. Phải là một trong: {valid_levels}")

# Load cấu hình mặc định
config = load_config()

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