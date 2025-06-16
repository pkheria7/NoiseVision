import cv2
import requests
import json
import time
from datetime import datetime
import os
from flask import Flask, request, jsonify
import numpy as np
import threading
from ultralytics import YOLO
import easyocr
from collections import deque
import queue
import pandas as pd
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
OUTPUT_DIR = "recordings"
LANE_DIR = "lane"  # Directory for lane-specific videos
DETECTED_DIR = "detected"  # Directory for detected license plates
CAMERA_ID = 0  # Default camera ID, change if needed
NUM_SECTIONS = 5  # Number of vertical sections
TARGET_FPS = 5  # Target frames per second
MAX_BUFFER_SIZE = 1000  # Maximum number of frames to keep in buffer

# Load dataset
try:
    dataset = pd.read_csv('dataset.csv')
    print("[DATASET] Successfully loaded dataset.csv")
except Exception as e:
    print(f"[DATASET] Error loading dataset: {str(e)}")
    dataset = pd.DataFrame()

def check_license_plate(plate_number):
    """Check if license plate exists in dataset and return associated data"""
    try:
        # Clean the plate number (remove spaces, convert to uppercase)
        plate_number = plate_number.strip().upper()
        
        # Search in dataset
        match = dataset[dataset['licence_plate'] == plate_number]
        
        if not match.empty:
            # Format phone number for Twilio (add +91 if not present)
            phone_number = match.iloc[0]['User Mobile Number']
            if not phone_number.startswith('+'):
                if not phone_number.startswith('91'):
                    phone_number = '+91' + phone_number
                else:
                    phone_number = '+' + phone_number
            
            return {
                'license_plate': match.iloc[0]['licence_plate'],
                'owner_name': match.iloc[0]['User Name'],
                'contact': phone_number
            }
        return None
    except Exception as e:
        print(f"[DATASET] Error checking license plate: {str(e)}")
        return None

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LANE_DIR, exist_ok=True)
os.makedirs(DETECTED_DIR, exist_ok=True)

# Initialize YOLO model and EasyOCR reader
model = YOLO('runs/detect/lpr_yolov8n/weights/best.pt')
reader = easyocr.Reader(['en'])

# Create frame buffer and processing queue
frame_buffer = deque(maxlen=MAX_BUFFER_SIZE)
frame_queue = queue.Queue()
processing_status = {"total_frames": 0, "processed_frames": 0, "detected_plates": 0}

class CameraService:
    def __init__(self):
        self.camera = None
        self.recording = False
        self.video_writer = None
        self.current_lane = None
        self.start_time = None
        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None
        self.section_width = None
        self.lock = threading.Lock()
        self.last_frame_time = None
        self.frame_interval = 1.0 / TARGET_FPS
        self.current_lane_dir = None
        self.initialize_camera()
        self.start_continuous_recording()

    def initialize_camera(self):
        """Initialize or reinitialize the camera"""
        if self.camera is not None:
            self.camera.release()
        self.camera = cv2.VideoCapture(CAMERA_ID)
        if not self.camera.isOpened():
            raise Exception("Failed to open camera")
        
        # Get frame dimensions
        ret, frame = self.camera.read()
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.section_width = self.frame_width // NUM_SECTIONS
            print(f"[CAMERA] Camera initialized successfully. Frame size: {self.frame_width}x{self.frame_height}")
            print(f"[CAMERA] Section width: {self.section_width} pixels")
        else:
            raise Exception("Failed to read initial frame")

    def start_continuous_recording(self):
        """Start continuous recording with the current lane"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # Create video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/continuous_recording_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create video writer for full frame width
        self.video_writer = cv2.VideoWriter(
            filename, 
            fourcc, 
            TARGET_FPS, 
            (self.frame_width, self.frame_height)  # Full frame width
        )
        
        self.recording = True
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.frame_count = 0
        print(f"[CAMERA] Started continuous recording at {TARGET_FPS} FPS")

    def start_lane_recording(self, lane_number):
        """Start a new lane-specific recording by creating a new directory"""
        # Create a new directory for this lane recording session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_lane_dir = os.path.join(LANE_DIR, f"lane_{lane_number}_{timestamp}")
        os.makedirs(self.current_lane_dir, exist_ok=True)
        print(f"[CAMERA] Started recording frames for Lane {lane_number} in {self.current_lane_dir}")

    def set_lane(self, lane_number):
        """Set the current lane for cropping"""
        with self.lock:
            if lane_number < 1 or lane_number > NUM_SECTIONS:
                raise ValueError(f"Invalid lane number: {lane_number}. Must be between 1 and {NUM_SECTIONS}")
            
            self.current_lane = lane_number
            # Start a new lane recording whenever lane changes
            self.start_lane_recording(lane_number)
            print(f"[CAMERA] Set to record Lane {lane_number}")

    def process_frame(self):
        """Process and record a single frame"""
        if self.recording and self.camera is not None:
            current_time = time.time()
            
            # Check if enough time has passed since last frame
            if current_time - self.last_frame_time < self.frame_interval:
                return False
                
            ret, frame = self.camera.read()
            if ret:
                with self.lock:
                    if self.current_lane is not None:
                        # Create a copy of the frame
                        processed_frame = frame.copy()
                        lane_frame = frame.copy()
                        
                        # Calculate lane boundaries
                        section_idx = NUM_SECTIONS - self.current_lane
                        start_x = section_idx * self.section_width
                        end_x = (section_idx + 1) * self.section_width
                        
                        # Calculate 10% expansion on both sides
                        expansion = int(self.section_width * 0.1)
                        
                        # Adjust start and end coordinates with expansion
                        expanded_start_x = max(0, start_x - expansion)
                        expanded_end_x = min(self.frame_width, end_x + expansion)
                        
                        # Blacken out left side
                        if expanded_start_x > 0:
                            processed_frame[:, :expanded_start_x] = 0
                        # Blacken out right side
                        if expanded_end_x < self.frame_width:
                            processed_frame[:, expanded_end_x:] = 0
                        
                        # Add lane number and timestamp to frame
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(processed_frame, f"Lane: {self.current_lane}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(processed_frame, timestamp, (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Draw green lines for section boundaries
                        cv2.line(processed_frame, (expanded_start_x, 0), (expanded_start_x, self.frame_height), (0, 255, 0), 2)
                        cv2.line(processed_frame, (expanded_end_x-1, 0), (expanded_end_x-1, self.frame_height), (0, 255, 0), 2)
                        
                        # For lane frame, crop to just the lane area
                        lane_frame = lane_frame[:, expanded_start_x:expanded_end_x]
                        
                        self.video_writer.write(processed_frame)
                        
                        # Save lane frame as image if we have a current lane directory
                        if self.current_lane_dir is not None:
                            frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            frame_filename = os.path.join(self.current_lane_dir, f"frame_{frame_timestamp}.jpg")
                            cv2.imwrite(frame_filename, lane_frame)
                            
                            # Add frame to buffer and queue for YOLO processing
                            frame_info = {
                                'frame': lane_frame,
                                'timestamp': frame_timestamp,
                                'lane': self.current_lane,
                                'filename': frame_filename
                            }
                            frame_buffer.append(frame_info)
                            frame_queue.put(frame_info)
                            processing_status["total_frames"] += 1
                        
                        self.frame_count += 1
                        self.last_frame_time = current_time
                        return True
            else:
                # If frame read fails, try to reinitialize camera
                print("[CAMERA] Frame read failed, attempting to reinitialize camera...")
                self.initialize_camera()
        return False

    def cleanup(self):
        """Clean up resources"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None

# Create global camera service instance
camera_service = CameraService()

@app.route('/camera', methods=['POST'])
def set_lane():
    try:
        data = request.get_json()
        lane_number = data.get('lane')
        if lane_number is None:
            return jsonify({"error": "No lane number provided"}), 400
        
        if lane_number < 1 or lane_number > NUM_SECTIONS:
            return jsonify({"error": f"Lane number must be between 1 and {NUM_SECTIONS}"}), 400
        
        camera_service.set_lane(lane_number)
        return jsonify({"status": "success", "message": f"Set to record lane {lane_number}"})
    except Exception as e:
        print(f"[CAMERA] Error in set_lane: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/camera/status', methods=['GET'])
def get_status():
    return jsonify({
        "recording": camera_service.recording,
        "lane": camera_service.current_lane,
        "frames_captured": camera_service.frame_count,
        "duration": time.time() - camera_service.start_time if camera_service.recording else 0,
        "section_width": camera_service.section_width if camera_service.section_width else None
    })

def process_frames():
    """Background thread to process video frames"""
    consecutive_failures = 0
    while True:
        try:
            if camera_service.recording:
                if not camera_service.process_frame():
                    consecutive_failures += 1
                    if consecutive_failures > 10:  # If we fail 10 times in a row
                        print("[CAMERA] Multiple frame read failures, reinitializing camera...")
                        camera_service.initialize_camera()
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0
            time.sleep(0.001)  # Small delay to prevent CPU overload
        except Exception as e:
            print(f"[CAMERA] Error in process_frames: {str(e)}")
            time.sleep(1)  # Wait a bit before retrying

def yolo_processing_thread():
    """Thread for processing frames with YOLO"""
    while True:
        try:
            # Get frame from queue
            frame_info = frame_queue.get()
            
            # Process frame with YOLO
            plate_text, plate_coords = detect_license_plate(frame_info['frame'])
            
            if plate_text:
                print(f"License plate detected in Lane {frame_info['lane']}: {plate_text}")
                processing_status["detected_plates"] += 1
                
                # Check if plate exists in dataset
                plate_data = check_license_plate(plate_text)
                # Create detected frame with plate information
                detected_frame = frame_info['frame'].copy()
                
                # Add plate text and lane number to the frame
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(detected_frame, timestamp, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw rectangle around the plate if coordinates are available
                if plate_coords:
                    x1, y1, x2, y2 = plate_coords
                    cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Save the detected frame
                detected_filename = os.path.join(DETECTED_DIR, 
                                               f"lane_{frame_info['lane']}_{frame_info['timestamp']}_{plate_text}.jpg")
                cv2.imwrite(detected_filename, detected_frame)
            
            processing_status["processed_frames"] += 1
            print(f"YOLO Processing Status: {processing_status['processed_frames']}/{processing_status['total_frames']} frames processed, {processing_status['detected_plates']} plates detected")
            
            # Mark task as done
            frame_queue.task_done()
            
        except Exception as e:
            print(f"Error in YOLO processing: {str(e)}")
            time.sleep(1)

def detect_license_plate(frame):
    """Detect license plate in the frame using YOLO and EasyOCR"""
    try:
        # Run YOLOv8 inference
        results = model(frame)[0]

        if len(results.boxes) == 0:
            return None, None

        # Get the highest confidence box
        boxes = results.boxes
        max_conf = 0
        best_plate = None

        for box in boxes:
            conf = float(box.conf)
            if conf > max_conf:
                max_conf = conf
                best_plate = box

        # Get coordinates and crop license plate
        x1, y1, x2, y2 = map(int, best_plate.xyxy[0].tolist())
        plate_img = frame[y1:y2, x1:x2]

        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # OCR using EasyOCR
        result = reader.readtext(gray)

        if result:
            plate_text = result[0][-2]
            return plate_text, (x1, y1, x2, y2)
        return None, None

    except Exception as e:
        print(f"Error in license plate detection: {str(e)}")
        return None, None

if __name__ == "__main__":
    import threading
    
    # Start frame processing thread
    frame_thread = threading.Thread(target=process_frames, daemon=True)
    frame_thread.start()
    
    # Start YOLO processing thread
    yolo_thread = threading.Thread(target=yolo_processing_thread, daemon=True)
    yolo_thread.start()
    
    try:
        print("[CAMERA] Starting camera service on port 5001...")
        print("[CAMERA] Press 'q' to stop recording")
        print("[CAMERA] Press 'r' to start recording")
        print("[CAMERA] Press 'x' to exit the program")
        
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5001))
        flask_thread.daemon = True
        flask_thread.start()
        
        # Wait for key press
        while True:
            key = input().lower()
            if key == 'q':
                print("\n[CAMERA] Stopping recording...")
                camera_service.recording = False
                if camera_service.video_writer is not None:
                    camera_service.video_writer.release()
                    camera_service.video_writer = None
                print("[CAMERA] Recording stopped. Server still running.")
            elif key == 'r':
                print("\n[CAMERA] Starting new recording...")
                camera_service.start_continuous_recording()
                print("[CAMERA] Recording started")
            elif key == 'x':
                print("\n[CAMERA] Shutting down...")
                break
                
    except KeyboardInterrupt:
        print("\n[CAMERA] Shutting down...")
    finally:
        camera_service.cleanup()
        print("[CAMERA] Service stopped")
