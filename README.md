
# Egyptian Vehicle License Plate Recognition System

## Project Overview
This project is a comprehensive system designed to recognize vehicle license plates from images and videos, process the data, and manage it through a backend service. It utilizes machine learning models for plate detection, Python for image and video processing, and a Node.js backend for data management.

## Components
- **Machine Learning Models**: Located in the `models/` directory, these are used for recognizing license plates from images and videos.
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
#
# `Important`
## Download Machine Learning Models
The machine learning models required for this project are stored on Google Drive. You can download them from [this link](https://drive.google.com/drive/folders/your-link-here). After downloading, place them in the `models/` directory.

if the linke is deprecated contact with me  to get the new link `tareqma7md@gmail.com`

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
Run `video.py` to process videos for license plate detection over time. Similar to image processing, ensure the backend service is running and set your own video or your camera id path at line `23`.

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

## Conclusion
This project demonstrates a full-stack approach to vehicle license plate detection and data management. By leveraging machine learning, Python, and Node.js, it offers a robust solution for processing and handling data from images and videos.
