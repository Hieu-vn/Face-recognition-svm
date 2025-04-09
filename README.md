# Face Recognition with InsightFace

## Mô tả
Hệ thống nhận diện khuôn mặt thời gian thực sử dụng InsightFace, hỗ trợ nhận diện nhiều người, log chi tiết, và xử lý lỗi.

## Cài đặt
1. Cài đặt các thư viện:
pip install -r requirements.txt

2. Chuẩn bị dữ liệu:
- Đặt ảnh vào `data/known_faces/` và `data/unknown_faces/`.

## Chạy hệ thống
1. Chạy toàn bộ: `python main.py`
2. Kiểm tra: `python -m unittest tests/*.py`

## Cấu hình
- Sửa `config/config.yaml` để điều chỉnh đường dẫn, ngưỡng, v.v.

## Yêu cầu
- Python 3.6+
- GPU (tùy chọn) với CUDA/CuDNN.