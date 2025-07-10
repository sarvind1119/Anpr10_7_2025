# YOLO-Based Vehicle Detection and Cropping

This project uses [YOLOv8](https://github.com/ultralytics/ultralytics) (You Only Look Once) for detecting vehicles (cars, buses, trucks) in video files.  
When a vehicle is detected, the script saves a cropped image of that vehicle into a folder named with the current date.  
Duplicate captures of the same vehicle are minimized by simple logic based on position and timing.

---

## Features

- **Detects cars, buses, and trucks in video footage** using YOLOv8.
- **Crops and saves each unique vehicle** as an image, organized by date.
- **Prevents duplicate images** for the same vehicle as it passes through the frame.
- **Easy to configure** for use with any video file.

---

## Installation

1. **Clone this repository** (or download the code files).

2. **Create a Python virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install required packages:**

    ```bash
    pip install ultralytics opencv-python
    ```

    > The `ultralytics` package includes YOLOv8 and all its dependencies.

---

## Usage

1. **Edit `app.py`** and set the path to your video file:

    ```python
    video_path = r"E:\July2025\Anpr10_7\Toll.mp4"   # Change as needed
    ```

2. **Run the script:**

    ```bash
    python app.py
    ```

3. **Output:**
    - Cropped images of vehicles will be saved in:
      ```
      detected_cars/YYYY-MM-DD/
      ```
      where `YYYY-MM-DD` is the current date.

4. **To exit:**  
   - The video window will display detections in real time.
   - Press `q` to quit early.

