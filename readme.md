  # Real-Time Object Detection using YOLOv4
# Project Overview
This project is a real-time object detection system developed for the Computer Vision Lab. It uses the YOLOv4 (You Only Look Once) deep learning model to identify objects from a live webcam feed.
The system processes video frames, identifies objects, and draws bounding boxes with class labels and confidence scores in real-time.

# Requirements
To run this project, you need the following Python libraries:
OpenCV: pip install opencv-python
NumPy: pip install numpy

# Project Structure
OBJECT DETECTION USING YOLVO.PY: The main Python execution script.
yolov4.cfg: The configuration file defining the network architecture.
coco.names: List of 80 object classes (person, car, etc.).
yolov4.weights: (Manual Download Required) The pre-trained model weights.

# Setup Instructions
Clone the Repository:
Download these files into a single folder on your local machine.

# Download Model Weights:
Due to GitHub file size limits (100MB+), the .weights file is not included in this repository.
Please download it from the official source:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
Place the downloaded file inside the same project folder.

# Run the Application:
Open your terminal/command prompt in the project folder and run:
python "OBJECT DETECTION USING YOLVO.PY"
# Controls
ESC: Press the Escape key to close the webcam window and stop the program.

Lab Objectives Met
Implemented a real-time deep learning pipeline.
Configured OpenCV DNN module for inference.
Applied Non-Maximum Suppression (NMS) to filter redundant detections.