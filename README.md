# NoiseVision: Vehicle Detection System

## Overview
NoiseVision is an innovative vehicle detection system that uses an array of I2S microphones to detect and track vehicles across multiple lanes. The system combines audio processing with computer vision to provide real-time vehicle detection and license plate recognition.

## Demo
https://github.com/pkheria7/NoiseVision/blob/master/demo.mp4

*Note: The demo video shows the system in action, demonstrating real-time vehicle detection and lane tracking using the microphone array.*

## System Architecture

### Hardware Components
1. **Microphone Array**
   - 3 I2S microphones arranged in a linear array
   - Each microphone is connected to a separate ESP32 microcontroller
   - Microphones are positioned at specific intervals (0cm, 20cm, 40cm) to create a 5-lane detection zone

2. **ESP32 Microcontrollers**
   - Each ESP32 handles one I2S microphone
   - Processes audio data at 22050Hz sampling rate
   - Communicates with Raspberry Pi via USB serial connection
   - Identified as MIC1, MIC2, and MIC3

3. **Raspberry Pi**
   - Acts as the central processing unit
   - Receives data from all three ESP32s
   - Processes audio data to determine vehicle position
   - Communicates with the Flask server

4. **Camera System**
   - Connected to the Flask server
   - Captures and processes video for license plate recognition
   - Uses YOLOv8 for license plate detection
   - Implements EasyOCR for text recognition

### Software Components

#### 1. ESP32 Firmware (last.py)
- Handles audio data acquisition from I2S microphones
- Implements real-time audio processing
- Features:
  - 128-frame buffer
  - 22050Hz sampling rate
  - Bandpass filtering (80Hz - 4000Hz)
  - Energy-based detection
  - Automatic device handshaking

#### 2. Raspberry Pi Processing (last.py)
- Manages communication with ESP32s
- Implements lane detection algorithm
- Features:
  - Real-time energy visualization
  - 5-lane detection system
  - Adjustable sensitivity and energy thresholds
  - Live visualization of microphone energy levels
  - Automatic camera triggering

#### 3. Flask Server (receiver.py)
- Handles camera control and video processing
- Implements license plate recognition
- Features:
  - YOLOv8-based license plate detection
  - EasyOCR text recognition
  - Video recording and storage
  - Lane-specific video segmentation
  - License plate database integration

## System Flow

1. **Audio Detection**
   - ESP32s continuously sample audio from I2S microphones
   - Audio data is streamed to Raspberry Pi
   - Energy levels are calculated for each microphone
   - System detects vehicle presence based on energy thresholds

2. **Lane Detection**
   - Raspberry Pi processes audio data from all microphones
   - Determines vehicle position using energy-based algorithm
   - Identifies which of the 5 lanes the vehicle is in
   - Triggers camera system when vehicle is detected

3. **Video Processing**
   - Camera captures video of detected vehicle
   - YOLOv8 model detects license plate
   - EasyOCR extracts license plate text
   - System checks plate against database
   - Stores video and detection data

## Configuration

### Audio Processing
- Sampling Rate: 22050Hz
- Buffer Size: 128 frames
- Bandpass Filter: 80Hz - 4000Hz
- Energy Threshold: Adjustable (default: 0.00003)
- Sensitivity: Adjustable (default: 1.0)

### Camera Settings
- Target FPS: 5
- Video Format: MP4
- Output Directories:
  - recordings/: Full video recordings
  - lane/: Lane-specific video segments
  - detected/: License plate detection images

## Usage

1. **Starting the System**
   ```bash
   # Start the Flask server
   python receiver.py

   # Start the Raspberry Pi processing
   python last.py
   ```

2. **Controls**
   - 'q': Stop recording
   - 'r': Start new recording
   - 'x': Exit program

3. **Visualization**
   - Real-time energy levels for each microphone
   - Lane detection visualization
   - Adjustable sensitivity and energy threshold sliders

## Requirements

### Hardware
- 3x ESP32 microcontrollers
- 3x I2S microphones
- Raspberry Pi
- Camera system
- USB connections

### Software
- Python 3.x
- OpenCV
- Flask
- NumPy
- SciPy
- Matplotlib
- PySerial
- Ultralytics YOLOv8
- EasyOCR

## License
[Add your license information here]

## Contributing
[Add contribution guidelines here]
