import cv2
import torch
from ultralytics import YOLO
from emotion import detect_emotion  # Hàm nhận diện cảm xúc
from repvgg import create_model
import numpy as np
from PIL import Image

# Kiểm tra thiết bị có GPU không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình YOLOv8 để phát hiện khuôn mặt
face_model = YOLO("weights/yolov8x-face.pt").to(device)

# Load mô hình nhận diện cảm xúc
emotion_model = create_model(device)
emotion_model.eval()  # Chuyển sang chế độ đánh giá

# Nhãn cảm xúc (8 loại, bao gồm "Contempt")
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Contempt"]

# Đọc ảnh
image_path = r"C:\Users\ADMIN\OneDrive\Desktop\YOLO\data_class\train\angry\image0000965_jpg.rf.9d013063184e3eb6fadbb5de9c5d7745.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Không thể mở ảnh!")
    exit()

# Chạy mô hình YOLO để phát hiện khuôn mặt
results = face_model(frame)

# Danh sách chứa khuôn mặt và bounding box
faces = []
boxes = []

for result in results:
    if result.boxes is not None:
        for box in result.boxes.xyxy:  # Lấy tọa độ bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)  # Chuyển tọa độ về integer
            face_crop = frame[y1:y2, x1:x2]  # Cắt ảnh khuôn mặt

            if face_crop.size == 0:
                continue  # Bỏ qua nếu ảnh khuôn mặt bị lỗi

            # Resize về 48x48 để khớp input của model cảm xúc
            face_resized = cv2.resize(face_crop, (48, 48))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
            
            # Chuyển đổi thành Tensor và chuẩn hóa
            face_tensor = torch.tensor(face_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            faces.append(face_tensor)
            boxes.append((x1, y1, x2, y2))

# Nếu phát hiện khuôn mặt, thực hiện nhận diện cảm xúc
if faces:
    faces_tensor = torch.cat(faces, dim=0)  # Ghép tất cả khuôn mặt thành batch

    with torch.no_grad():
        outputs = emotion_model(faces_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # Chuyển đổi thành xác suất
        confidence, emotion_idx = torch.max(probabilities, dim=1)  # Lấy nhãn có xác suất cao nhất

    # Vẽ bounding box và nhãn cảm xúc lên frame
    for (x1, y1, x2, y2), idx, conf in zip(boxes, emotion_idx, confidence):
        label = f"{EMOTION_LABELS[idx.item()]} {conf.item() * 100:.1f}%"  # Ví dụ: "Angry 95.2%"
        color = (0, 255, 0)  # Màu xanh lá cây cho bounding box
        
        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Hiển thị nhãn cảm xúc trên khung hình
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size

        # Đảm bảo vị trí text không bị tràn khỏi ảnh
        text_x = x1
        text_y = max(y1 - 10, text_h + 10)  # Tránh bị tràn lên trên

        # Vẽ nền chữ màu đen
        cv2.rectangle(frame, (text_x, text_y - text_h - 5), (text_x + text_w + 10, text_y + 5), (0, 0, 0), -1)

        # Hiển thị chữ màu trắng
        cv2.putText(frame, label, (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Hiển thị kết quả
cv2.imshow("Emotion Detection", frame)
cv2.waitKey(0)  # Đợi nhấn phím bất kỳ để đóng cửa sổ
cv2.destroyAllWindows()
