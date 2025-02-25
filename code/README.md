# Emotion Detection with YOLO
This project uses YOLO model for emotion detection. The model is built with PyTorch, and the code integrates several libraries such as OpenCV, Ultralyitcs YOLO, RepVGG, and others to handle image processing and emotion classification.

## Requirements
- Python 3.11 - 3.13
- The following dependencies need to be installed:

### Install library (cmd)
pip install -r requirements.txt

#### Example code to test model
import cv2
from emotion import detect_emotion  
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
img = cv2.imread("path_to_image.jpg")
result = detect_emotion(img)
print(result)
