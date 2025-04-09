import logging
import os
from src.config_manager import load_config, get_config_value

def setup_logger():
    """
    Khởi tạo logger cho dự án nhận diện khuôn mặt.

    - Tạo logger chính với tên 'face_recognition'.
    - Thiết lập 2 handler: một cho thông tin (INFO) và một cho lỗi (ERROR).
    - Lưu log vào file (info.log và error.log) và hiển thị trên console.

    Returns:
        logging.Logger: Logger đã được cấu hình.
    """
    # Đọc cấu hình
    config = load_config()
    enable_logging = get_config_value(config, 'logging', 'enable', True)
    log_level = get_config_value(config, 'logging', 'level', "INFO")
    error_log_file = get_config_value(config, 'paths', 'error_log_file', "../logs/error.log")
    info_log_file = get_config_value(config, 'paths', 'info_log_file', "../logs/info.log")

    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
    os.makedirs(os.path.dirname(info_log_file), exist_ok=True)

    # Tạo logger
    logger = logging.getLogger('face_recognition')
    logger.setLevel(logging.DEBUG if enable_logging else logging.CRITICAL)

    # Xóa các handler cũ (tránh trùng lặp nếu setup lại)
    logger.handlers.clear()

    # Định dạng log
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler cho file info.log (INFO trở lên)
    info_handler = logging.FileHandler(info_log_file)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(log_format)
    logger.addHandler(info_handler)

    # Handler cho file error.log (ERROR trở lên)
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    logger.addHandler(error_handler)

    # Handler cho console (dựa trên log_level từ config)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

# Khởi tạo logger mặc định
logger = setup_logger()

if __name__ == "__main__":
    # Kiểm tra logger
    logger.debug("Đây là thông điệp DEBUG.")
    logger.info("Đây là thông điệp INFO.")
    logger.warning("Đây là thông điệp WARNING.")
    logger.error("Đây là thông điệp ERROR.")
    logger.critical("Đây là thông điệp CRITICAL.")