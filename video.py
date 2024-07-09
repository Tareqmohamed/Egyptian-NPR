import os
import cv2
from ultralytics import YOLO
import requests
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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

# Parameters
video_path = "/dev/video2"
camera_id = "camera_1"  # Example camera ID
frame_interval = 10  # Interval in frames to process

# Initialize the main window
root = tk.Tk()
root.title("Car Plate Detection")

# Create a frame for video display
video_frame = ttk.Frame(root)
video_frame.grid(row=0, column=0, padx=10, pady=10)

# Create a label for displaying the video
video_label = ttk.Label(video_frame)
video_label.grid(row=0, column=0)

# Create a frame for detected plate display
plate_frame = ttk.Frame(root)
plate_frame.grid(row=0, column=1, padx=10, pady=10)

# Create a label for displaying the detected plate image
plate_image_label = ttk.Label(plate_frame)
plate_image_label.grid(row=0, column=0)

# Create a label for displaying the detected plate number
plate_number_label = ttk.Label(plate_frame, font=("Helvetica", 24))
plate_number_label.grid(row=1, column=0)

# Create a frame for buttons
button_frame = ttk.Frame(root)
button_frame.grid(row=1, column=0, padx=10, pady=10, columnspan=2)

# Create a frame for text box
textbox_frame = ttk.Frame(root)
textbox_frame.grid(row=2, column=0, padx=10, pady=10, columnspan=2)

# Create a text box to display detected plate numbers
textbox = tk.Text(textbox_frame, height=10, width=50)
textbox.grid(row=0, column=0, padx=5, pady=5)

# Variables for capturing and detection states
is_capturing = False
is_detecting = False

# List to store detected plates
detected_plates = []

# Function to start capturing video
def start_capture():
    global is_capturing, cap, frame_count
    if not is_capturing:
        is_capturing = True
        cap = cv2.VideoCapture(video_path)
        start_capture_button.config(text="Stop Capturing", command=stop_capture)
        frame_count = 0
        capture_video()

# Function to stop capturing video
def stop_capture():
    global is_capturing
    if is_capturing:
        is_capturing = False
        cap.release()
        start_capture_button.config(text="Start Capturing", command=start_capture)

# Function to capture video frames
def capture_video():
    global frame_count
    if not is_capturing:
        return  # Stop capturing if the flag is not set
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        frame_count += 1
        if is_detecting and frame_count % frame_interval == 0:
            detect_plate(frame_rgb)
    root.after(10, capture_video)

# Function to start detection
def start_detection():
    global is_detecting
    if not is_detecting:
        is_detecting = True
        start_detection_button.config(text="Stop Detection", command=start_detection)
    else:
        is_detecting = False
        start_detection_button.config(text="Start Detection", command=start_detection)

# Function to detect car plate and characters
def detect_plate(frame_rgb):
    global detected_plates, is_detecting
    plate_detection = plate_model.track(frame_rgb, save_txt=False, conf=.4, device="cuda")
    for plate in plate_detection:
        car_boxes = plate.boxes
        if len(car_boxes) != 0:
            x1, y1, x2, y2 = car_boxes.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_plate = frame_rgb[y1:y2, x1:x2]

            ocr_result = ocr_model.predict(cropped_plate, conf=0.01, iou=0.1, max_det=7, device="cuda")

            detected_chars = []
            for en_chr in ocr_result:
                for box in en_chr.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_chars.append({"class": to_arabic(str(box.cls.item())), "x": x1, "y": y1})

            detected_chars = sorted(detected_chars, key=lambda x: x["x"])

            plate_number = ""
            for char in detected_chars:
                plate_number += char["class"] + " "

            plate_number = ''.join(reversed(plate_number.strip()))

            if plate_number in detected_plates:
                continue

            detected_plates.append(plate_number)

            plate_dir = os.path.join("./out/", plate_number)
            if not os.path.exists(plate_dir):
                os.makedirs(plate_dir)

            frame_path = os.path.join(plate_dir, f"frame_{frame_count}.jpg")
            plate_path = os.path.join(plate_dir, "plate.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(plate_path, cv2.cvtColor(cropped_plate, cv2.COLOR_RGB2BGR))

            print(f"Detected plate number: {plate_number}")
            print(f"Saved frame and plate to {plate_dir}")

            capture_date = datetime.now().isoformat()
            payload = {
                "plateNumber": plate_number,
                "captureDate": capture_date,
                "cameraId": camera_id,
                # "platePath": cv2.imread(plate_path),
                # "carPath": cv2.imread(frame_path),
                "roads": "alex highway",
                
            }
            datasend=""
            response = requests.put("https://localhost:3000/api/videos/plates", json=payload)
            if response.status_code == 200:
                datasend="Data successfully sent to backend"
                print(datasend)
            else:
                print(f"Failed to send data to backend: {response.status_code} {response.text}")

            # Draw rectangle around the detected plate on the frame
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Update the GUI with the detected plate image and number
            plate_img = Image.fromarray(cropped_plate)
            plate_imgtk = ImageTk.PhotoImage(image=plate_img)
            plate_image_label.imgtk = plate_imgtk
            plate_image_label.configure(image=plate_imgtk)

            plate_number_label.config(text=f"Detected Plate Number: {plate_number}")

            # Append detected plate number with date to text box, clearing previous content
            textbox.delete(1.0, tk.END)
            textbox.insert(tk.END, f"{capture_date}   -     {plate_number}\n  {datasend}")
            textbox.config(font=("Helvetica", 24))

            # Stop detection after first detection
            is_detecting = False
            start_detection_button.config(text="Start Detection", command=start_detection)
            break

# Create buttons to start/stop capturing and detection
start_capture_button = ttk.Button(button_frame, text="Start Capturing", command=start_capture)
start_capture_button.grid(row=0, column=0, padx=5, pady=5)

start_detection_button = ttk.Button(button_frame, text="Start Detection", command=start_detection)
start_detection_button.grid(row=0, column=1, padx=5, pady=5)

# Run the Tkinter event loop
root.mainloop()
