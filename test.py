import cv2

video_path = r"E:\July2025\Anpr10_7\Toll.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video file!")
else:
    print("Video file opened successfully!")
    ret, frame = cap.read()
    if ret:
        print("Read first frame successfully.")
        cv2.imshow("First Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to read first frame.")

cap.release()
