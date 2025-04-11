import cv2
import numpy as np
import onnxruntime as ort
import os
import sys

# Set input video path
video_path = "traffic_sample.mp4"
if not os.path.exists(video_path):
    print(f"[ERROR] Video file not found: {video_path}")
    sys.exit(1)

# Load YOLOv5 ONNX model
model_path = "yolov5s.onnx"
if not os.path.exists(model_path):
    print(f"[ERROR] ONNX model not found: {model_path}")
    sys.exit(1)

# Initialize ONNX Runtime session
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # (1, 3, H, W)
input_height, input_width = input_shape[2], input_shape[3]

def preprocess(frame):
    img = cv2.resize(frame, (input_width, input_height))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW format
    return np.expand_dims(img, axis=0)  # Add batch dim

def postprocess(outputs, img_shape, conf_threshold=0.4):
    preds = outputs[0]  # Shape: (1, num_detections, 85)
    preds = np.squeeze(preds)  # -> (num_detections, 85)

    boxes = []
    h, w = img_shape

    for pred in preds:
        objectness = pred[4]
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        confidence = objectness * class_scores[class_id]

        # Vehicle classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        if confidence > conf_threshold and class_id in [2, 3, 5, 7]:
            x_center, y_center, width, height = pred[0:4]
            x1 = int((x_center - width / 2) * w / input_width)
            y1 = int((y_center - height / 2) * h / input_height)
            x2 = int((x_center + width / 2) * w / input_width)
            y2 = int((y_center + height / 2) * h / input_height)
            boxes.append((x1, y1, x2, y2, y_center))  # Save y_center for tracking

    return boxes

# Open video capture
cap = cv2.VideoCapture(video_path)

# Get video width and height to ensure the window is correctly sized
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a named window with the correct size
cv2.namedWindow('Traffic Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Traffic Detection', width, height)

# Original FPS (from video) and cap it to 30
original_fps = cap.get(cv2.CAP_PROP_FPS)
fps = min(original_fps, 30.0)
delta_time = 1.0 / fps

print(f"[INFO] Original video FPS: {original_fps}")
print(f"[INFO] Capped FPS for processing: {fps:.2f}")

prev_frame_y = []
vehicle_ids = []  # To store vehicle ID and its position for tracking

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    boxes = postprocess(outputs, frame.shape[:2])

    curr_frame_y = [(x1, y1, x2, y2, y_center) for (x1, y1, x2, y2, y_center) in boxes]
    curr_frame_vehicle_ids = []

    # Debug: Check how many vehicles are detected in the current frame
    print(f"[INFO] Detected vehicles: {len(curr_frame_y)}")

    # Track vehicles across frames
    if prev_frame_y:
        # Match vehicles from previous and current frames
        matched_vehicles, unmatched_vehicles = match_vehicles(vehicle_ids, curr_frame_y)

        # Assign new IDs to unmatched vehicles
        next_vehicle_id = max([id for id, _ in vehicle_ids], default=-1) + 1
        for _, vehicle in unmatched_vehicles:
            curr_frame_vehicle_ids.append((next_vehicle_id, vehicle))
            next_vehicle_id += 1

        # Update the previous frame's vehicle positions
        vehicle_ids = matched_vehicles + curr_frame_vehicle_ids
    else:
        # On the first frame, initialize vehicle tracking with detected vehicles
        next_vehicle_id = 0
        for vehicle in curr_frame_y:
            curr_frame_vehicle_ids.append((next_vehicle_id, vehicle))
            next_vehicle_id += 1
        vehicle_ids = curr_frame_vehicle_ids

    # Write vehicle positions to positions.txt (now it will be written every frame with detected vehicles)
    if curr_frame_y:
        print(f"[INFO] Writing vehicle positions to positions.txt...")
        with open("positions.txt", "a") as f:  # Use 'a' to append to the file
            for vehicle_id, (x1, y1, x2, y2, y_center) in vehicle_ids:
                f.write(f"{vehicle_id} {y_center} {delta_time}\n")
    else:
        print(f"[INFO] No vehicles detected in this frame.")

    # Display frame with detections and vehicle count
    for (x1, y1, x2, y2, _) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Vehicles: {len(curr_frame_y)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:  # ESC to exit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()