from ultralytics import YOLO
import cv2
import os
from datetime import datetime

model = YOLO('yolov8n.pt')
# Set up video source (0 for webcam or path to video file)
video_path = r"E:\\July2025\\Anpr10_7\\Toll.mp4"
cap = cv2.VideoCapture(video_path)

today_str = datetime.now().strftime('%Y-%m-%d')
output_dir = os.path.join('detected_cars', today_str)
os.makedirs(output_dir, exist_ok=True)

img_count = 0
last_detection_time = {}
labels = {2: 'car', 5: 'bus', 7: 'truck'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_time = datetime.now().timestamp()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in labels:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2)//2, (y1 + y2)//2)
                recently_seen = False
                for pos, t in last_detection_time.items():
                    if abs(center[0] - pos[0]) < 50 and abs(center[1] - pos[1]) < 50 and (current_time - t) < 3:
                        recently_seen = True
                        break
                if not recently_seen:
                    img_count += 1
                    timestamp = datetime.now().strftime('%H%M%S_%f')[:-3]
                    label = labels[cls]
                    filename = f"{label}_{timestamp}_{img_count}.jpg"
                    car_img = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(output_dir, filename), car_img)
                    last_detection_time[center] = current_time
                # Clean up old entries
                last_detection_time = {k: v for k, v in last_detection_time.items() if current_time - v < 10}
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, labels[cls], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('YOLOv8 Vehicle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
