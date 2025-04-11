
# Real-time Traffic Analysis with CUDA Object Detection

This project performs real-time traffic analysis using CUDA for vehicle detection and speed estimation. The system processes a video input, detects vehicles, estimates their speeds, and visualizes the results. The analysis includes several key visualizations for insights into vehicle speeds, variations, and distributions.

---

## Project Overview

The **Real-time Traffic Analysis with CUDA Object Detection** project focuses on detecting vehicles in a video stream using YOLOv5 for object detection and CUDA for accelerating the speed estimation of the detected vehicles. The results are visualized in several insightful plots that allow an understanding of traffic conditions in a specific area.

### Key Features:
- **Vehicle Detection**: Detect vehicles in video frames using YOLOv5 for real-time detection.
- **Speed Estimation**: Estimate vehicle speeds based on bounding box displacements over frames using CUDA for acceleration.
- **Visualization**: Provides various plots such as histograms, speed variation over time, box plots, and more.
- **Real-time Processing**: Process each frame of the video in real-time to estimate speeds and update visualizations.

---

## Tech Stack

- **Programming Languages**: 
  - Python (for vehicle detection, speed estimation, and visualization)
  - C++ (for CUDA-based speed estimation)
  
- **CUDA**: 
  - Acceleration of vehicle speed estimation using GPU capabilities for faster computation.

- **Deep Learning**: 
  - **YOLOv5 (ONNX format)**: Used for detecting vehicles in each frame of the video.

- **Libraries & Tools**:
  - **Python**:
    - **PyTorch**: For running YOLOv5 models.
    - **OpenCV**: For handling video frames.
    - **NumPy**: For numerical calculations.
    - **Matplotlib/Seaborn**: For plotting the results and generating visualizations.
  - **CUDA**: For parallel processing of speed estimation using GPU acceleration.
  
---

## Installation

### 1. Clone the Repository:
```bash
git clone https://github.com/<your-username>/Real-time-Traffic-Analysis-with-CUDA-Object-Detection.git
cd Real-time-Traffic-Analysis-with-CUDA-Object-Detection
```

### 2. Install Dependencies:
- First, make sure you have **CUDA** installed (along with compatible NVIDIA drivers).
- You will also need **Python 3.8+** and **pip** for the Python environment.

#### Install Python dependencies:
```bash
pip install -r requirements.txt
```

- The `requirements.txt` file contains all the necessary dependencies like `torch`, `opencv-python`, `matplotlib`, `seaborn`, and others required for the project.

---

## Usage

1. **Run the Detection and Speed Estimation Script**:
   - The project processes video input (`traffic_sample.mp4`) and estimates vehicle speeds. Make sure your input video and other required files are placed in the project directory.
   - Execute the CUDA-based speed estimation with the following command:

   ```bash
   python traffic_detection.py
   ```

   This will use the **YOLOv5 ONNX model** to detect vehicles in the video frames, outputting the vehicle positions in `positions.txt` and bounding box data in `bounding_boxes.txt`.

2. **Run Visualization Script**:
   - After running the detection script, use the following Python script to generate plots from the speed data:

   ```bash
   python Visualize_results.py
   ```

   This will process the `output_speeds.txt` file and visualize various metrics, including histograms, box plots, and speed variations.

---

## Visualizations

- **Histogram**: Distribution of average vehicle speeds across all detected vehicles.
- **Speed Variation Over Time**: A plot showing how the speed of individual vehicles varies over time (frames).
- **Box Plot**: A summary of the distribution of average vehicle speeds.
- **Top 10 Slowest/Fastest Vehicles**: A bar chart displaying the slowest and fastest vehicles based on average speeds.
- **Violin Plot**: Distribution of speeds for each vehicle in the dataset.

---

## Project Structure

Python Project
```
├── traffic_detection.py        # YOLOv5 detection and speed estimation (CUDA)
├── Visualize_results.py        # Visualization script for speed data
├── traffic_sample.mp4          # Sample input video for traffic analysis
├── output_speeds.txt           # Vehicle speed data for visualization
├── positions.txt               # Vehicle positions in each frame
├── requirements.txt            # Python dependencies
```

CUDA Project
```
├── output_speeds.txt           # Vehicle speed data for visualization
├── positions.txt               # Vehicle positions in each frame
├── kernel.cu                   # CUDA kernel for speed estimation
└── other_project_files/         # Any additional files
```

---

## How It Works

1. **Vehicle Detection**:
   - The **YOLOv5** model (converted to ONNX format) is used for detecting vehicles in each frame of the video.
   - Each detected vehicle's bounding box is recorded for subsequent processing.

2. **Speed Estimation**:
   - A CUDA-based kernel (`kernel.cu`) computes the displacement of bounding boxes between consecutive frames to estimate vehicle speeds in pixels per frame (px/frame).

3. **Visualization**:
   - The script `Visualize_results.py` generates visualizations such as histograms, box plots, and bar charts to provide insights into vehicle speeds and traffic conditions.

---
