import cv2
from ultralytics import YOLO
import datetime
import os

# Create the folder if it doesn't exist
folder_name = "init_state"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize camera (index 0 is usually the default)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print(f"Press 'q' to capture a picture into '{folder_name}' and get data.")
print("Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Capture and analyze when 'q' is pressed
    if key == ord('q'):
        # Save the picture in the init_state folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder_name, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"\n--- Captured: {filename} ---")

        # Run detection
        results = model(frame)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # Get position (xyxy format: x1, y1, x2, y2)
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = coords
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                print(f"Class: {cls_name}")
                print(f"  Position (Box): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"  Center: ({cx:.1f}, {cy:.1f})")
                print(f"  Confidence: {box.conf[0]:.2f}")
        
        print("----------------------------\n")

    # Exit on ESC
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
