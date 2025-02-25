import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import torchvision.models as models

# Danh sách các nhãn cảm xúc
EMOTION_LABELS = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def load_emotion_model(model_path="weights/emotion_model.pth", device="cpu"):
    """ Load mô hình nhận diện cảm xúc với trọng số đã huấn luyện """
    model = models.resnet18(weights=None)  # Khởi tạo mô hình trống
    model.fc = torch.nn.Linear(model.fc.in_features, len(EMOTION_LABELS))  # 8 lớp cảm xúc

    # Tải trọng số đã huấn luyện
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Chuyển sang chế độ đánh giá (không train)

    return model

# Hàm tiền xử lý ảnh khuôn mặt
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize ảnh về kích thước 48x48 (phù hợp với model)
    transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh sang grayscale
    transforms.ToTensor(),  # Chuyển thành tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Chuẩn hóa
])

def preprocess_image(img):
    if isinstance(img, Image.Image):  # Nếu img là PIL Image, chuyển sang NumPy array
        img = np.array(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh về grayscale
    img = cv2.resize(img, (128, 128))  # Resize về đúng kích thước phù hợp (kiểm tra với model)
    img = img.astype("float32") / 255.0  # Chuẩn hóa về [0,1]

    img = np.expand_dims(img, axis=0)  # Thêm kênh → (1, 128, 128)
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension → (1, 1, 128, 128)

    return torch.tensor(img, dtype=torch.float32)


def detect_emotion(images, model, device, show_conf=False):
    """ Dự đoán cảm xúc từ danh sách ảnh khuôn mặt """
    results = []
    for img in images:
        img_tensor = preprocess_image(img).to(device)  # Chuyển ảnh về Tensor

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            label_idx = torch.argmax(probs).item()
            confidence = probs[0, label_idx].item()

        label = EMOTION_LABELS[label_idx]
        if show_conf:
            label = f"{label} ({confidence:.2f})"

        results.append((label, label_idx))
    return results
