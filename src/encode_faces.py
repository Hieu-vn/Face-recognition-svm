import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from src.config_manager import get_config_value, load_config
from src.path_manager import path_manager
from src.model_manager import model_manager
from src.image_processor import ImageProcessor
from utils.logger_init import logger

def encode_faces(input_dir=None, use_original=True):
    """
    Trích xuất embeddings từ ảnh khuôn mặt
    
    Args:
        input_dir (str, optional): Thư mục chứa ảnh khuôn mặt.
                                 Nếu None, sẽ sử dụng thư mục mặc định từ path_manager.
        use_original (bool): Nếu True, sẽ sử dụng ảnh gốc thay vì ảnh đã xử lý.
    
    Returns:
        bool: True nếu thành công, False nếu thất bại.
    """
    try:
        # Lấy đường dẫn từ path_manager
        if input_dir is None:
            # Nếu sử dụng ảnh gốc, lấy từ thư mục known_faces thay vì known_faces_processed
            if use_original:
                input_dir = path_manager.get_path('known_faces')
            else:
                input_dir = path_manager.get_path('known_faces_processed')
        
        output_file = path_manager.get_path('encodings_file')
        
        # Kiểm tra thư mục đầu vào
        if not os.path.exists(input_dir):
            logger.error(f"Thư mục {input_dir} không tồn tại")
            return False
            
        # Khởi tạo ImageProcessor và đảm bảo model InsightFace đã được khởi tạo
        if model_manager.face_app is None:
            model_manager._initialize_face_app()
            if model_manager.face_app is None:
                logger.error("Không thể khởi tạo InsightFace model")
                return False
        
        # Lấy danh sách tất cả các file ảnh
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(Path(input_dir).glob(f'**/{ext}')))
        
        if not image_files:
            logger.error(f"Không tìm thấy ảnh nào trong thư mục {input_dir}")
            return False
            
        logger.info(f"Tìm thấy {len(image_files)} ảnh trong thư mục {input_dir}")
        
        # Lấy cấu hình
        config = load_config()
        target_size = tuple(get_config_value(config, 'image_processing', 'target_size', [112, 112]))
        
        # Khởi tạo danh sách embeddings và tên
        embeddings = []
        names = []
        
        # Khởi tạo ImageProcessor
        image_processor = ImageProcessor()
        
        # Xử lý từng ảnh
        for img_path in tqdm(image_files, desc="Trích xuất embeddings"):
            try:
                # Đọc ảnh
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.error(f"Không thể đọc ảnh: {img_path}")
                    continue
                
                # Kiểm tra độ phân giải và kích thước ảnh
                h, w = image.shape[:2]
                file_size_kb = os.path.getsize(str(img_path)) / 1024
                logger.debug(f"Ảnh {img_path.name}: độ phân giải {w}x{h}, kích thước {file_size_kb:.2f}KB")
                
                # Lấy tên người từ tên thư mục hoặc tên file
                # Lấy tên từ cấu trúc thư mục: data/known_faces/person_name/image.jpg
                name = img_path.parent.name
                if name == Path(input_dir).name or name == "processed":  
                    # Nếu ảnh nằm trực tiếp trong thư mục đầu vào, sử dụng tên file
                    name = img_path.stem.split('_')[0]  # Lấy phần đầu tên file trước dấu gạch dưới
                
                # 1. Phát hiện và cắt khuôn mặt sử dụng ImageProcessor cải tiến
                face_img, face_obj = image_processor.detect_and_crop_face(image, str(img_path))
                
                if face_img is not None:
                    # Nếu phát hiện được khuôn mặt, lấy embedding
                    embedding = model_manager.get_embedding(face_img)
                    if embedding is not None:
                        embeddings.append(embedding)
                        names.append(name)
                        logger.debug(f"Đã trích xuất embedding cho {name}")
                        continue  # Tiếp tục với ảnh tiếp theo
                
                # 2. Nếu không phát hiện được khuôn mặt hoặc không trích xuất được embedding
                # Thử phát hiện khuôn mặt trực tiếp với ảnh gốc
                
                # Thử các kích thước khác nhau
                test_sizes = [(640, 640), (480, 480), (720, 720), (1080, 1080)]
                
                # Kiểm tra kích thước ảnh gốc, nếu quá nhỏ thì tăng kích thước
                min_dim = min(h, w)
                if min_dim < 300:
                    # Ảnh quá nhỏ, tăng kích thước lên
                    scale = 300 / min_dim
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    logger.debug(f"Đã resize ảnh từ {w}x{h} lên {new_w}x{new_h}")
                    h, w = new_h, new_w
                
                # Trực tiếp sử dụng model.face_app.get mà KHÔNG sử dụng tham số threshold (gây lỗi)
                for test_size in test_sizes:
                    try:
                        # Resize ảnh về kích thước phù hợp
                        if max(h, w) > max(test_size):
                            scale = min(test_size[0]/w, test_size[1]/h)
                            dim = (int(w*scale), int(h*scale))
                            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                        else:
                            resized = image
                            
                        # Đặt lại det_size (điều chỉnh kích thước cửa sổ phát hiện)
                        model_manager.face_app.prepare(ctx_id=-1, det_size=test_size)
                        
                        # Gọi get() không có tham số threshold
                        faces = model_manager.face_app.get(resized)
                        
                        if len(faces) > 0:
                            # Chọn khuôn mặt lớn nhất
                            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                            face = faces[0]
                            
                            # Lấy embedding
                            if hasattr(face, 'embedding') and face.embedding is not None:
                                embeddings.append(face.embedding)
                                names.append(name)
                                logger.debug(f"Đã trích xuất embedding cho {name} với kích thước {test_size}")
                                break
                            else:
                                # Cắt khuôn mặt và thử lấy embedding
                                bbox = face.bbox.astype(int)
                                x1, y1, x2, y2 = bbox
                                
                                # Điều chỉnh tọa độ về ảnh gốc nếu đã resize
                                if max(h, w) > max(test_size):
                                    scale_back = max(h, w) / max(test_size)
                                    x1 = int(x1 * scale_back)
                                    y1 = int(y1 * scale_back)
                                    x2 = int(x2 * scale_back)
                                    y2 = int(y2 * scale_back)
                                
                                # Đảm bảo tọa độ hợp lệ
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(w, x2)
                                y2 = min(h, y2)
                                
                                # Kiểm tra kích thước khuôn mặt có đủ lớn không
                                face_width, face_height = x2-x1, y2-y1
                                if face_width < 60 or face_height < 60:
                                    logger.warning(f"Khuôn mặt phát hiện quá nhỏ ({face_width}x{face_height})")
                                    continue
                                
                                # Mở rộng vùng khuôn mặt thêm một chút
                                padding = 0.1
                                width, height = x2-x1, y2-y1
                                x1 = max(0, x1 - int(padding * width))
                                y1 = max(0, y1 - int(padding * height))
                                x2 = min(w, x2 + int(padding * width))
                                y2 = min(h, y2 + int(padding * height))
                                
                                # Cắt và resize khuôn mặt
                                face_crop = image[y1:y2, x1:x2]
                                face_resize = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LANCZOS4)
                                
                                # Trích xuất embedding
                                embedding = model_manager.get_embedding(face_resize)
                                if embedding is not None:
                                    embeddings.append(embedding)
                                    names.append(name)
                                    logger.debug(f"Đã trích xuất embedding cho {name} bằng cách cắt khuôn mặt")
                                    break
                    except Exception as test_error:
                        logger.debug(f"Lỗi khi thử kích thước {test_size}: {test_error}")
                
                # Reset lại det_size về giá trị mặc định
                model_manager.face_app.prepare(ctx_id=-1, det_size=(640, 640))
                
                # Nếu vẫn không tìm được embedding
                if len(embeddings) != len(names) or (len(names) > 0 and names[-1] != name):
                    logger.warning(f"Không thể trích xuất embedding cho {name} sau khi thử tất cả phương pháp")
                    # Lưu ảnh lỗi để debug
                    debug_dir = Path(output_file).parent / "debug" / "failed_embeddings"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_file = debug_dir / f"failed_{img_path.name}"
                    cv2.imwrite(str(debug_file), image)
                    
            except Exception as e:
                logger.error(f"Lỗi khi xử lý ảnh {img_path}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
        # Kiểm tra số lượng embeddings
        if not embeddings:
            logger.error("Không trích xuất được embedding nào")
            return False
            
        # Chuyển đổi sang numpy array
        embeddings = np.array(embeddings)
        
        # Lưu embeddings
        if model_manager.save_encodings(embeddings, names):
            logger.info(f"Đã lưu {len(embeddings)} embeddings vào {output_file}")
            return True
        else:
            logger.error("Lỗi khi lưu encodings")
            return False
        
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất embeddings: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

if __name__ == "__main__":
    encode_faces()