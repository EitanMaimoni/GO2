from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
import sys

# Load YOLO model for person detection (use YOLOv4-tiny for better performance)
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# Handle different OpenCV versions for getUnconnectedOutLayers()
unconnected_out_layers = net.getUnconnectedOutLayers()
if unconnected_out_layers.ndim == 2:  # OpenCV returns a 2D array
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
else:  # OpenCV returns a 1D array
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Camera parameters (update these based on your camera)
fov = 70  # Field of view in degrees (approximate, adjust based on your camera)
known_person_height = 1.7  # Average person height in meters
focal_length = 600  # Focal length in pixels (approximate, adjust based on your camera)

def detect_person(image):
    height, width, channels = image.shape
    person_detected = False
    latest_distance = 0.0
    latest_angle = 0.0

    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # Reduced resolution
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.75 and classes[class_id] == 'person':
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Draw bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate distance and angle
                latest_distance = (known_person_height * focal_length) / h
                offset_x = center_x - (width / 2)
                latest_angle = (offset_x / (width / 2)) * (fov / 2)

                # Set person detected flag
                person_detected = True

    return person_detected, latest_distance, latest_angle, image

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = VideoClient()  # Create a video client
    client.SetTimeout(3.0)
    client.Init()

    code, data = client.GetImageSample()

    # Create a named window and resize it
    window_name = "front_camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
    cv2.resizeWindow(window_name, 800, 600)  # Set the window size to 800x600 pixels

    # Request normal when code==0
    while code == 0:
        try:
            # Get Image data from Go2 robot
            code, data = client.GetImageSample()

            # Convert to numpy image
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            # Check if the image is valid
            if image is not None:
                # Detect person and calculate distance/angle
                person_detected, distance, angle, image = detect_person(image)

                # Display the image with distance and angle
                height, width, _ = image.shape
                text_x = width // 2 - 100
                text_y = height - 50

                # Clear previous text by drawing a filled rectangle
                cv2.rectangle(image, (text_x - 10, text_y - 30), (text_x + 200, text_y + 10), (0, 0, 0), -1)

                # Display distance and angle
                cv2.putText(image, f"Distance: {distance:.2f} m", (text_x, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f"Angle: {angle:.2f} deg", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display image
                cv2.imshow(window_name, image)
                # Press ESC to stop
                if cv2.waitKey(20) == 27:
                    break
            else:
                print("Received bad image, ignoring...")

        except Exception as e:
            print(f"Error processing image: {e}. Ignoring bad image...")
            continue

    if code != 0:
        print("Get image sample error. code:", code)
    else:
        # Capture an image
        cv2.imwrite("front_image.jpg", image)

    cv2.destroyWindow(window_name)