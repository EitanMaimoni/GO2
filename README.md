# GO2 Project Setup Guide

## Step 1: Install and Configure the Official SDK

Before starting with the GO2 project, you first need to install the **Unitree Robotics SDK**. Follow these steps:

1. Visit the [Unitree SDK GitHub page](https://github.com/unitreerobotics/unitree_sdk2_python).
2. Download and install the SDK following the instructions provided on the repository.

### **Important Note:**
For the SDK to work properly, you need to manually configure your **IPv4 interface** (wired connection) to use a specific IP address. Set the IP address to something like `192.168.123.222` and the subnet mask to `255.255.255.0` (mask 24). This configuration is necessary for communication with the robot.

Once the SDK is working correctly, proceed to the next step.

---

## Step 2: Install Required Packages

After installing and confirming the SDK works, you can proceed with setting up the necessary dependencies for the GO2 project.

### **Install System Dependencies:**
Run the following commands to install essential system packages:

```bash
sudo apt update
sudo apt install -y libgtk2.0-dev pkg-config
```

### **Install OpenCV:**
Next, install OpenCV using `pip`:

```bash
pip install opencv-python
```

---

## Step 3: Download YOLO Files

If the YOLO files are missing, you need to download them from the official YOLO website. Ensure that you download the following files:

- **YOLOv4 Weights:** [Download yolov4.weights](https://github.com/AlexeyAB/darknet/releases)
- **YOLOv4 Configuration:** [Download yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)
- **COCO Names:** [Download coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

### **Place the YOLO Files in the Correct Directory**

1. **Create a new directory called `model`** in the same location as the `src` directory.
2. **Move the downloaded YOLO files** into the newly created `model` directory.
   - The structure should look like this:

   ```
   /GO2
   ├── /src
   ├── /model
   │   ├── yolov4.weights
   │   ├── yolov4.cfg
   │   └── coco.names
   ```

By ensuring these files are placed in the `model` directory, the project will be able to locate them correctly when running the script.

---

## Step 4: Running the Project

Once all dependencies are installed and the YOLO files are downloaded, you can run the project using the following command:

```bash
python3 main.py "en0(hash)"
```

Make sure that all the necessary files (such as the YOLO weights, configuration, and class names) are placed in the correct directories before executing the script.

---

## Summary

1. Install and configure the **Unitree SDK** and ensure it’s working.
2. Install required system dependencies and OpenCV.
3. Download the YOLO files and place them in the correct directories.
4. Run the project with `python3 main.py "en0(hash)"`.

If you encounter any issues, make sure to check that all files are correctly placed and that your SDK is properly configured.
