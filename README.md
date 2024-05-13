# Sequre: AI-Powered Surveillance System

## Overview
Sequre is an advanced, AI-powered surveillance application designed to enhance security systems through real-time object detection. Utilising the powerful YOLOv8 model, Sequre analyses video footage to intelligently identify and monitor specified objects within user-defined regions of interest (ROIs). Developed in Python, this tool integrates sophisticated computer vision techniques using OpenCV and NumPy, ensuring efficient and accurate processing of video feeds.

## Features
- **Real-Time Object Detection:** Detect specified objects within video footage with high accuracy and minimal latency.
- **Dynamic ROIs:** Users can define and adjust ROIs on a per-video basis for focused monitoring.
- **Video Quality Handling:** Efficiently processes various video qualities and formats.
- **RTSP Support:** Seamlessly operates with RTSP video feeds.
- **Customisable Monitoring:** Allows for the monitoring of multiple object classes as per user requirements.

## Future Endeavours
- **Suspicious Activity Detection:** Enhance the system to detect suspicious activities such as loitering or suspicious characters within the ROI.
- **User Interface Enhancements:** Develop a more intuitive and user-friendly interface.
- **Integration with IoT Devices:** Enable integration with IoT devices for automated responses.
- **Extended Object Classification:** Broaden the range of detectable objects for comprehensive monitoring.

## Installation
To install and set up Sequre, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/aidenparkerr/sequre.git
    cd sequre
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the YOLOv8 model weights:**  
    Follow the instructions on the [YOLOv8 repository](https://github.com/ultralytics/yolov8) to download the pretrained weights and save them in the `models/` directory.

## Usage
1. **Configure the application:**
    - Edit the configuration file at `configs/config.json` to set your video source, model path, and other parameters.

2. **Run the application:**
    ```bash
    python main.py --config configs/config.json
    ```

3. **Set the ROI:**
    - On the first run, you will be prompted to set the region of interest by selecting points on the video frame.

## Directory Structure
```plaintext
sequre/
├── configs/
│   └── config.json
├── data/
│   ├── input/
│   └── output/
├── models/
│   └── yolov8.pt
├── src/
│   ├── data_handler.py
│   ├── detector.py
│   ├── main.py
│   └── utils/
│       ├── config_reader.py
│       ├── file_utils.py
│       ├── logger.py
│       └── visualise.py
└── README.md
