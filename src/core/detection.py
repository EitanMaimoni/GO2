import cv2
import numpy as np
import math

class PersonDetector:
    """
    A class for detecting persons in images using YOLO object detection.
    """
    
    def __init__(self, settings):
        """
        Initialize the YOLO person detector.
        
        Args:
            settings: Application settings object
        """
        # Load YOLO model
        self.net = cv2.dnn.readNet(settings.yolo_weights, settings.yolo_cfg)
        with open(settings.yolo_names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        if unconnected_out_layers.ndim == 2:  # OpenCV returns a 2D array
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:  # OpenCV returns a 1D array
            self.output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

        # Camera parameters for distance/angle estimation
        self.robot_params = settings.robot_params
        self.confidence_threshold = settings.detection_confidence
        self.target_capture_size = settings.target_capture_size

    def detect_persons(self, image):
        """
        Detect all persons in an image using YOLO and apply NMS.
        
        Returns:
            list: List of dictionaries with 'box', 'distance', 'angle'
        """
        if image is None:
            return []

        height, width = image.shape[:2]
        detections = []

        # Prepare input blob
        blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold and self.classes[class_id] == 'person':
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)

        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            x, y, w, h = boxes[i]
            x, y, w, h = self._validate_bbox(x, y, w, h, width, height)
            if w <= 0 or h <= 0:
                continue

            x_adjusted = x - (width / 2)
            y_adjusted = height - (y + h)

            distance = self._calculate_distance(x_adjusted, y_adjusted)
            angle = self._calculate_angle(x + w / 2, width)

            detections.append({
                'box': (x, y, w, h),
                'distance': distance,
                'angle': angle
            })

        return detections


    def get_first_person(self, image):
        """
        Detect and return the first person found in the input image.
        
        Args:
            image: Input image (numpy array)
                
        Returns:
            tuple: (cropped_image, detection_info) or (None, None) if no person is detected.
                - cropped_image: Cropped image of the detected person
                - detection_info:
                    Dictionary containing detection info:
                    - 'box': (x, y, w, h) bounding box coordinates
                    - 'distance': Estimated distance to person in meters
                    - 'angle': Estimated angle to person in degrees
        """
        if image is None:
            return None, None
            
        detections = self.detect_persons(image)
        
        if not detections:
            return None, None
            
        # Get the first detection
        detection = detections[0]
        x, y, w, h = detection['box']
        
        # Crop the person
        person_img = image[y:y+h, x:x+w]
        if person_img.size == 0:
            return None, None

        # Resize to target size (force exact dimensions)
        try:
            resized_img = cv2.resize(person_img, self.target_capture_size, 
                                interpolation=cv2.INTER_LINEAR)
            return resized_img, detection
        except:
            return None, None

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
    
    def _calculate_distance(self, x, y):
        """
        Calculate distance to the person based on the y-coordinate.
        
        Args:
            x: X-coordinate of the bounding box center
            y: Y-coordinate of the bounding box center
            
        Returns:
            float: Estimated distance in meters
        """
        # Calculate distance using the Pythagorean theorem and scaling the result
        return math.sqrt(x**2 + y**2) / 100


    def _calculate_angle(self, center_x, img_width):
        """
        Calculate angle to the person based on the x-coordinate.
        
        Args:
            center_x: X-coordinate of the bounding box center
            img_width: Width of the image
            
        Returns:
            float: Estimated angle in degrees
        """
        return ((center_x - (img_width / 2)) / (img_width / 2)) * (self.robot_params.camera_fov / 2)