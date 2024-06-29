
# Egyptian Vehicle License Plate Recognition System

## Project Overview
This project is a comprehensive system designed to recognize vehicle license plates from images and videos, process the data, and manage it through a backend service. It utilizes machine learning models for plate detection, Python for image and video processing, and a Node.js backend for data management.

## Components
- **Machine Learning Models**: Located in the `models/` directory, these are used for recognizing license plates from images and videos. YOLO (You Only Look Once) is employed for plate detection and OCR, offering real-time processing and high accuracy.
- **Image Processing**: Handled by `img.py` and `img.ipynb`, these scripts use the machine learning models to detect license plates in images.
- **Video Processing**: Managed by `video.py`, this script processes video files to detect license plates over time.
- **Backend Service**: Located in the `backend-test/` directory, this Node.js application receives and manages data related to detected license plates.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.11.2 or higher
- Node.js and npm
- Required Python libraries as specified in `requirements.txt`
- Required Node.js packages as specified in `backend-test/package.json`
- **CUDA** (if using GPU for processing)

## Why YOLO for Plate Detection and OCR?
YOLO (You Only Look Once) is chosen for license plate detection and OCR because of its numerous advantages, making it a standout choice for real-time applications such as this project. Here’s a detailed look at why YOLO excels:

### Speed
- **Real-Time Processing**: YOLO is designed for real-time applications. It can process images and videos at high speeds, making it possible to detect license plates almost instantaneously as the data is received. This is crucial for applications like traffic monitoring and surveillance, where timely detection is essential.
- **Low Latency**: The architecture of YOLO allows it to process an image in a single pass, reducing the latency compared to models that require multiple passes or stages.

### Accuracy
- **High Precision**: YOLO achieves high accuracy in detecting objects by dividing the image into a grid and predicting bounding boxes and class probabilities for each grid cell. This method ensures that even small objects, like license plates, are detected reliably.
- **Fewer False Positives**: YOLO’s approach minimizes the occurrence of false positives (incorrect detections), which is critical for applications where precision is important, such as reading license plates correctly for enforcement or toll collection.

### Efficiency
- **Resource Management**: YOLO is designed to be computationally efficient, making it suitable for deployment on a variety of hardware, from powerful GPUs to less powerful CPUs. This flexibility is beneficial for scaling the application across different environments, from edge devices to cloud servers.
- **Optimized Performance**: YOLO’s single-shot detection mechanism means it can handle multiple detections within an image in a streamlined and efficient manner. This is in contrast to models that might need to process parts of the image multiple times, consuming more resources.

### Single Pass Detection
- **Simplified Pipeline**: YOLO performs detection in a single pass through the network. It predicts multiple bounding boxes and class probabilities in one evaluation, significantly simplifying the detection pipeline. This single-pass approach contributes to its speed and efficiency.
- **Unified Detection**: By predicting all bounding boxes and class probabilities simultaneously, YOLO avoids the need for separate region proposal networks or complex post-processing steps that other models might require. This unification helps in maintaining high processing speeds.

### Additional Advantages
- **Robustness**: YOLO is robust to various types of input data, whether it be images taken under different lighting conditions, angles, or resolutions. This robustness ensures consistent performance in diverse real-world scenarios.
- **Scalability**: YOLO can be scaled to different versions (e.g., YOLOv3, YOLOv4, YOLOv5) that trade off between speed and accuracy, allowing developers to choose the most appropriate version for their specific use case.
- **Community and Support**: YOLO has a strong community and a wealth of resources available, making it easier to implement, customize, and troubleshoot. This support is invaluable for maintaining and updating the system over time.
By leveraging YOLO, this project benefits from a cutting-edge object detection framework that is fast, accurate, and efficient, making it ideal for real-time license plate recognition tasks.
  
## Installation

### Clone the Repository
Start by cloning the repository to your local machine:
```bash
git clone https://github.com/Tareqmohamed/Egyptian-NPR.git
cd Egyptian-NPR
```

### Install Python Dependencies
Navigate to the project root and run:
```bash
pip install -r requirements.txt
```

### Install Node.js Dependencies
Navigate to the `backend-test/` directory and run:
```bash
cd backend-test
npm install
```

### Verify CUDA Installation
Ensure that CUDA is installed on your system. You can check this by running:
```bash
nvcc --version
```
If CUDA is not installed, it is preferred to install it. Follow these steps:

1. **Install NVIDIA Driver**:
    ```bash
    sudo apt update
    sudo apt install nvidia-driver-525
    ```

2. **Download and Install CUDA Toolkit**:
    Visit the [CUDA Toolkit Download Page](https://developer.nvidia.com/cuda-downloads) and select your operating system, architecture, distribution, and version. Follow the instructions provided.

    Example for Ubuntu:
    ```bash
    sudo dpkg -i cuda-repo-<distro>_<version>_amd64.deb
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda
    ```

3. **Set Environment Variables**:
    Add the CUDA paths to your `.bashrc` or `.zshrc`:
    ```bash
    export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```

    Replace `11.8` with the version of CUDA you installed.

4. **Apply the changes**:
    ```bash
    source ~/.bashrc
    ```

If you do not wish to install CUDA, you can modify the script to use the CPU instead. Open `video.py` and remove `device="cuda"` at line `45` and line `76` in `img.py`.

## Download Machine Learning Models
The machine learning models required for this project are stored on Google Drive. You can download them from [this link](https://drive.google.com/drive/folders/105UFyY4EdW1IpY8XWdhlZ3ToCvBNPR0A?usp=sharing). After downloading, place them in the `models/` directory.

If the link is deprecated, contact me to get the new link: `tareqma7md@gmail.com`.

## Running the Project

### Backend Service
1. Navigate to the `backend-test/` directory.
2. Start the server using npm:
```bash
npm start
```
This will start the backend service on port 3000.

### Image Processing
Run `img.py` or use the Jupyter notebook `img.ipynb` to process images for license plate detection. Ensure the backend service is running to handle the data and set your own image path at line `51`.

### Video Processing
Run `video.py` to process videos for license plate detection over time. Similar to image processing, ensure the backend service is running and set your own video or your camera ID path at line `23`.

## Project Structure
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `backend-test/`: Contains the Node.js backend service.
- `img.ipynb`: Jupyter notebook for image processing.
- `img.py`: Python script for image processing.
- `models/`: Contains machine learning models for license plate detection (not included in the repository).
- `photos/`: Directory for storing photos (not included in the repository).
- `requirements.txt`: Lists the Python dependencies.
- `train/`: Contains training data for machine learning models (ignored by Git).
- `video.py`: Python script for video processing.
- `videos/`: Directory for storing videos.

## Data Management
The backend service receives data related to detected license plates, including plate number, capture date, camera ID, and file paths for the plate and car images. This data is logged and can be extended for further processing or integration with databases.



By leveraging YOLO, this project ensures robust and reliable license plate recognition, which is critical for real-time applications such as surveillance and traffic monitoring.

## Conclusion
This project demonstrates a full-stack approach to vehicle license plate detection and data management. By leveraging machine learning, Python, and Node.js, it offers a robust solution for processing and handling data from images and videos.
