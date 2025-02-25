import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

# ======= Định nghĩa mô hình RepVGG =======
class RepVGGBlock(nn.Module):
    """ Khối RepVGG giúp tăng tốc độ suy luận """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(RepVGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class RepVGG(nn.Module):
    def __init__(self, num_classes=8):
        super(RepVGG, self).__init__()
        self.layer1 = RepVGGBlock(1, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = RepVGGBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.layer3 = RepVGGBlock(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling xuống kích thước (1,1)
        self.fc = nn.Linear(256, num_classes)  # Cập nhật kích thước phù hợp

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)  # Kích thước (batch, 256, 1, 1)
        x = torch.flatten(x, 1)  # Chuyển thành (batch, 256)
        return self.fc(x)


def create_model(device="cpu"):
    """ Hàm tạo mô hình """
    model = RepVGG(num_classes=8).to(device)
    return model


# ======= Hàm tiền xử lý ảnh =======
def preprocess_image(img):
    """
    Chuyển đổi ảnh đầu vào thành tensor phù hợp với mô hình
    """
    if isinstance(img, Image.Image):  # Nếu img là PIL Image, chuyển sang NumPy array
        img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh về grayscale
    img = cv2.resize(img, (64, 64))  # Resize về đúng kích thước phù hợp
    img = img.astype("float32") / 255.0  # Chuẩn hóa về [0,1]

    img = np.expand_dims(img, axis=0)  # Thêm kênh → (1, 64, 64)
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension → (1, 1, 64, 64)

    return torch.tensor(img, dtype=torch.float32)
