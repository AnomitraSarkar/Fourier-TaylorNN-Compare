# Install required libraries if not already installed
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary libraries
try:
    import torch
    import cv2
except ImportError:
    install("torch")
    install("torchvision")
    install("opencv-python")

# Clone YOLOv5 repository if not already cloned
import os

if not os.path.exists("yolov5"):
    print("Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

# Import required libraries
import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load image using OpenCV
image_path = 'image.png'  # Replace with your image path
img = cv2.imread(image_path)

# Check if the image is loaded properly
if img is None:
    print("Failed to load image. Check the path.")
    exit()

# Convert BGR (OpenCV) to RGB (YOLO expects RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model(img_rgb)

# Print detected objects
print("\nDetection Results:")
print(results.pandas().xyxy[0])  # Pandas DataFrame with detection results

# Extract and display results
for idx, row in results.pandas().xyxy[0].iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    conf = row['confidence']
    label = row['name']

    # Draw bounding boxes
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show image with detections
cv2.imshow("YOLOv5 Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
