# Position Detection and Verification System

This project uses YOLOv8 for real-time object detection and position verification. It allows you to set a reference "preset" position for an object and then verify if the object is in the correct spot during a live feed.

## Files and Structure

- **`capture_data.py`**: Used to set the initial reference state.
- **`verify_position.py`**: Used for live verification against the reference state.
- **`init_state/`**: A folder created automatically to store reference images.
- **`yolov8n.pt`**: The YOLOv8 Nano model weights used for detection.

---

## 1. Initial State Capture (`capture_data.py`)

This script is used to capture the "correct" or "ideal" position of an object.

### How it works:
1. Opens your camera feed.
2. When you press **'q'**:
   - It captures the current frame.
   - Saves the image into the `init_state` folder with a timestamp (e.g., `capture_20240307_123456.jpg`).
   - Runs YOLOv8 detection on the captured frame.
   - Prints the detected object class, bounding box coordinates, and center point to the console.
3. Press **'ESC'** to exit the camera feed.

---

## 2. Live Position Verification (`verify_position.py`)

This script compares the live camera feed against the latest captured reference image.

### How it works:
1. **Reference Loading**: It automatically finds the latest image in the `init_state` folder and detects the object within it to establish a "Preset" position.
2. **Live Detection**: It opens a live camera feed and continuously runs YOLOv8 detection.
3. **Visual Feedback**:
   - **Green Box**: Shows the "Preset" (reference) position.
   - **Red Box**: Shows the "Current" live position of the object.
   - **Cyan Line**: Draws a line connecting the centers of the preset and live objects.
4. **Calculations**:
   - It calculates the pixel difference between the live center and the reference center.
   - Converts the pixel difference to millimeters (mm) using a constant (`MM_PER_PIXEL = 0.4`).
5. **Status Messages**:
   - Displays the required movement (X and Y in mm) to reach the preset position.
   - Shows **"MATCHED"** in green if the object is within 2mm of the target and the class matches.
   - Warns if there is a **"Class Mismatch"** or if the object is out of position.

### Controls:
- Press **'ESC'** to quit the live verification.

---

## Requirements
- `opencv-python`
- `ultralytics` (YOLOv8)
- `numpy`
