import cv2
import numpy as np

class PersonDetector:
    def __init__(self, weights_path, cfg_path, names_path):
        # Load YOLO model
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        if unconnected_out_layers.ndim == 2:  # OpenCV returns a 2D array
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:  # OpenCV returns a 1D array
            self.output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

        # Camera parameters
        self.fov = 70  # Field of view in degrees
        self.known_person_height = 1.7  # Average person height in meters
        self.focal_length = 600  # Focal length in pixels

    def detect_person(self, image):
        height, width, _ = image.shape
        person_detected = False
        latest_distance = 0.0
        latest_angle = 0.0

        # Detect objects using YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.75 and self.classes[class_id] == 'person':
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
                    latest_distance = (self.known_person_height * self.focal_length) / h
                    offset_x = center_x - (width / 2)
                    latest_angle = (offset_x / (width / 2)) * (self.fov / 2)

                    # Set person detected flag
                    person_detected = True

        return person_detected, latest_distance, latest_angle, image