import cv2
import numpy as np

class PersonDetector:
    """
    A class for detecting persons in images using YOLO object detection.
    
    Responsibilities:
    1. Detect all persons in an image and return the image with bounding boxes drawn
    2. Return cropped images of detected persons
    
    Note: This class only handles detection, not person recognition/identification.
    """
    
    def __init__(self, weights_path, cfg_path, names_path):
        """
        Initialize the YOLO person detector.
        
        Args:
            weights_path: Path to YOLO weights file
            cfg_path: Path to YOLO config file
            names_path: Path to file containing class names
        """
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

        # Camera parameters for distance/angle estimation
        self.fov = 70  # Field of view in degrees
        self.known_person_height = 1.7  # Average person height in meters
        self.focal_length = 600  # Camera focal length in pixels

    def detect_persons(self, image):
        """
        Detect all persons in an image and return the image with bounding boxes drawn.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            tuple: (processed_image, detections)
            - processed_image: Input image with bounding boxes drawn around detected persons
            - detections: List of dictionaries containing detection info for each person:
                - 'box': (x, y, w, h) bounding box coordinates
                - 'distance': Estimated distance to person in meters
                - 'angle': Estimated angle to person in degrees
        """
        height, width = image.shape[:2]
        processed_image = image.copy()
        detections = []

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
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Validate bounding box
                    x, y, w, h = self._validate_bbox(x, y, w, h, width, height)
                    if w <= 0 or h <= 0:
                        continue

                    # Draw bounding box
                    cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate distance and angle
                    distance = (self.known_person_height * self.focal_length) / h
                    angle = ((center_x - (width / 2)) / (width / 2)) * (self.fov / 2)

                    # Store detection info
                    detections.append({
                        'box': (x, y, w, h),
                        'distance': distance,
                        'angle': angle
                    })

        return processed_image, detections

    def get_first_person(self, image, target_size=(128*4, 256*4)):
        """
        Detect and return the first person found in the input image.
        
        Args:
            image: Input image (numpy array)
            target_size: Tuple (width, height) specifying the size to resize the cropped image to
                
        Returns:
            Cropped and resized person image (numpy array) or None if no person is detected
        """
        height, width = image.shape[:2]

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
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Validate bounding box
                    x, y, w, h = self._validate_bbox(x, y, w, h, width, height)
                    if w <= 0 or h <= 0:
                        continue

                    # Crop the person
                    person_img = image[y:y+h, x:x+w]
                    if person_img.size == 0:
                        continue

                    # Resize to target size (force exact dimensions)
                    try:
                        resized_img = cv2.resize(person_img, target_size, 
                                            interpolation=cv2.INTER_LINEAR)
                        return resized_img  # Return immediately after finding the first person
                    except:
                        continue

        return None  # Return None if no person is detected

    def _validate_bbox(self, x, y, w, h, img_width, img_height):
        """
        Ensure bounding box coordinates are within image boundaries.
        
        Args:
            x, y: Top-left corner coordinates
            w, h: Width and height of bounding box
            img_width, img_height: Dimensions of the image
            
        Returns:
            Validated coordinates (x, y, w, h)
        """
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        return x, y, w, h