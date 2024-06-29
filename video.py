import os
import cv2
from ultralytics import YOLO
import requests
from datetime import datetime

# Function to convert detected character indices to Arabic characters
def to_arabic(en_chr):
    arabic_chars = {
        "9.0": "أ", "10.0": "ع", "11.0": "ب", "12.0": "د", "13.0": "ف",
        "14.0": "ج", "15.0": "ه", "16.0": "ك", "17.0": "ل", "18.0": "م",
        "19.0": "ن", "20.0": "ر", "21.0": "س", "22.0": "ص", "23.0": "ط",
        "24.0": "و", "25.0": "ى", "0.0": "١", "1.0": "٢", "2.0": "٣",
        "3.0": "٤", "4.0": "٥", "5.0": "٦", "6.0": "٧", "7.0": "٨", "8.0": "٩"
    }
    return arabic_chars.get(en_chr, None)

# Load YOLO models
plate_model = YOLO("./models/polo.pt")
ocr_model = YOLO("./models/best.pt")

# Path to video
video_path = "/dev/video0"
cap = cv2.VideoCapture(video_path)

# Parameters
frame_interval = .2 * int(cap.get(cv2.CAP_PROP_FPS))  # 60 seconds interval
camera_id = "camera_1"  # Example camera ID

# List to store detected plates
detected_plates = []

# Process video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect license plate
        plate_detection = plate_model(frame_rgb, save_txt=False, conf=0.1, device="cuda")

        # Crop and display detected plate
        for plate in plate_detection:
            car_boxes = plate.boxes
            if len(car_boxes) != 0:
                x1, y1, x2, y2 = car_boxes.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cropped_plate = frame_rgb[y1:y2, x1:x2]

                # Detect characters on the plate
                ocr_result = ocr_model.predict(cropped_plate, conf=0.2, iou=0.1, max_det=7, device="cuda")

                # Translate detected characters to Arabic
                detected_chars = []
                for en_chr in ocr_result:
                    for box in en_chr.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detected_chars.append({"class": to_arabic(str(box.cls.item())), "x": x1, "y": y1})

                # Sort characters by their x-coordinate
                detected_chars = sorted(detected_chars, key=lambda x: x["x"])

                # Combine characters into a single string
                plate_number = ""
                for char in detected_chars:
                    plate_number +=char["class"] + " "
                
                plate_number = ''.join(reversed(plate_number.strip()))

                # Check if the plate number is already detected
                if plate_number in detected_plates:
                    continue  # Skip this plate if it's already detected

                # Add the detected plate number to the list
                detected_plates.append(plate_number)

                # Create directory for the current plate number
                plate_dir = os.path.join("./out/", plate_number)
                if not os.path.exists(plate_dir):
                    os.makedirs(plate_dir)

                # Save the frame and cropped plate image
                frame_path = os.path.join(plate_dir, f"frame_{frame_count}.jpg")
                plate_path = os.path.join(plate_dir, "plate.jpg")
                cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(plate_path, cv2.cvtColor(cropped_plate, cv2.COLOR_RGB2BGR))

                print(f"Detected plate number: {plate_number}")
                print(f"Saved frame and plate to {plate_dir}")

                # Send data to the Node.js backend
                capture_date = datetime.now().isoformat()
                payload = {
                    "plateNumber": plate_number,
                    "captureDate": capture_date,
                    "cameraId": camera_id,
                    "platePath": plate_path,
                    "carPath": frame_path,
                    
                }
                response = requests.post("http://localhost:3000/videos/plates", json=payload)
                if response.status_code == 200:
                    print("Data successfully sent to backend")
                else:
                    print(f"Failed to send data to backend: {response.status_code} {response.text}")

    frame_count += 1

cap.release()
