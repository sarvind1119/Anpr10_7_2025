from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load YOLOv8 pretrained model (COCO dataset)
model = YOLO('yolov8s.pt')  # or 'yolov8s.pt' for more accuracy

# Set up video source (0 for webcam or path to video file)
video_path = r"E:\\July2025\\Anpr10_7\\Toll.mp4"
cap = cv2.VideoCapture(video_path)

today_str = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('detected_cars', today_str)
os.makedirs(output_dir, exist_ok=True)

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            # YOLO COCO class 2 is 'car', 5 is 'bus', 7 is 'truck', etc.
            if cls in [2, 5, 7]:  # Detect cars, buses, trucks
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_img = frame[y1:y2, x1:x2]
                img_count += 1
                timestamp = datetime.now().strftime('%H%M%S_%f')[:-3]
                filename = f"car_{timestamp}_{img_count}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), car_img)
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Car Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
