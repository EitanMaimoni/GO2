
 # GO2 Project
 
 ## Download YOLO Files
 If the YOLO files are missing, download them from the official YOLO website:
 
 - **YOLOv4 Weights**: [Download yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
 - **YOLOv4 Configuration**: [Download yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)
 - **COCO Names**: [Download coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
 
 ## Install Required Packages
 Run the following commands to install the necessary dependencies:
 
 ```bash
 sudo apt update
 sudo apt install -y libgtk2.0-dev pkg-config
 ```
 
 Then, install OpenCV:
 
 ```bash
 pip install opencv-python
 ```
 
 ## Running the Project
 Once dependencies are installed and YOLO files are downloaded, you can run the project using:
 
 ```bash
 python3 main.py "en0(hash)"
 ```
 
 Ensure that all necessary files are placed in the correct directories before executing the script.
