import cv2
from ultralytics import YOLO
import os
import glob

# Constants
FOLDER_NAME = "init_state"
MM_PER_PIXEL = 0.4  # Adjust based on your setup

# 1. Load the YOLO model
model = YOLO("yolov8n.pt")

def get_reference_data():
    """Finds the latest image in init_state and returns the detection box and center."""
    list_of_files = glob.glob(os.path.join(FOLDER_NAME, "*.jpg"))
    if not list_of_files:
        print(f"Error: No images found in '{FOLDER_NAME}' folder.")
        return None, None
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Using reference image: {latest_file}")
    
    ref_img = cv2.imread(latest_file)
    results = model(ref_img)
    
    if len(results[0].boxes) > 0:
        # Get the first detected object as the reference
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return box, (cx, cy)
    
    print("Error: No objects detected in the reference image.")
    return None, None

# 2. Get reference position
ref_box, ref_center = get_reference_data()

if ref_box is None:
    exit()

ref_x1, ref_y1, ref_x2, ref_y2 = ref_box
ref_cx, ref_cy = ref_center

# 3. Initialize Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Starting live verification...")
print("The green box is the PRESET position.")
print("The red box is the CURRENT position.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the PRESET position (Green Box)
    cv2.rectangle(frame, (int(ref_x1), int(ref_y1)), (int(ref_x2), int(ref_y2)), (0, 255, 0), 2)
    cv2.putText(frame, "Preset Position", (int(ref_x1), int(ref_y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Run detection on live frame
    results = model(frame)

    if len(results[0].boxes) > 0:
        # Find the object in the live frame (using the first one detected)
        live_box = results[0].boxes.xyxy[0].cpu().numpy()
        lx1, ly1, lx2, ly2 = live_box
        lcx = (lx1 + lx2) / 2
        lcy = (ly1 + ly2) / 2

        # Draw the CURRENT position (Red Box)
        cv2.rectangle(frame, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 0, 255), 2)

        # Calculate difference
        dx_pixel = lcx - ref_cx
        dy_pixel = lcy - ref_cy

        dx_mm = dx_pixel * MM_PER_PIXEL
        dy_mm = dy_pixel * MM_PER_PIXEL

        # Display movement info
        status_text = f"Move X: {dx_mm:+.2f}mm, Y: {dy_mm:+.2f}mm"
        color = (0, 255, 255) # Yellow
        
        if abs(dx_mm) < 2 and abs(dy_mm) < 2:
            status_text = "MATCHED: In correct position"
            color = (0, 255, 0) # Green

        cv2.putText(frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw a line between current and reference center
        cv2.line(frame, (int(ref_cx), int(ref_cy)), (int(lcx), int(lcy)), (255, 255, 0), 2)

    cv2.imshow("Live Verification", frame)

    if cv2.waitKey(1) == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
