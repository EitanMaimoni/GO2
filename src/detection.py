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
        self.fov = 70
        self.known_person_height = 1.7
        self.focal_length = 600

    def detect_person(self, image):
        """Simplified detection function that matches your usage pattern"""
        height, width, _ = image.shape
        person_detected = False
        distance = 0.0
        angle = 0.0
        processed_image = image.copy()

        # Detect objects
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
                    # Get bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Draw box
                    cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calculate metrics (store last detection's values)
                    distance = (self.known_person_height * self.focal_length) / h
                    angle = ((center_x - (width / 2)) / (width / 2)) * (self.fov / 2)
                    person_detected = True

        return person_detected, distance, angle, processed_image

    def get_cropped_persons(self, image, target_size=(128*4, 256*4)):
        """
        Get cropped person images and FORCE resize them to fixed dimensions
        (will stretch/squash if needed)
        Args:
            image: Input image
            target_size: Tuple (width, height) for output size
        Returns:
            List of resized person images
        """
        height, width, _ = image.shape
        cropped_images = []

        # Detect objects
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
                    # Get bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Validate crop coordinates
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w <= 0 or h <= 0:  # Skip invalid crops
                        continue

                    # Crop the person
                    person_img = image[y:y+h, x:x+w]
                    
                    if person_img.size == 0:  # Skip empty images
                        continue

                    # FORCE RESIZE (stretch/squash to exact size)
                    try:
                        resized_img = cv2.resize(person_img, target_size, interpolation=cv2.INTER_LINEAR)
                        cropped_images.append(resized_img)
                    except:
                        continue

        return cropped_images

    def _validate_bbox(self, x, y, w, h, img_width, img_height):
        """Ensure bounding box stays within image boundaries"""
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        return x, y, w, h

    def _resize_with_aspect(self, image, target_size):
        """
        Resize image while maintaining aspect ratio
        Returns None if input is invalid
        """
        if image is None or image.size == 0:
            return None

        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # Handle potential division by zero
        if w == 0 or h == 0:
            return None
        
        # Calculate ratio and new dimensions
        ratio = min(target_w/w, target_h/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        try:
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create blank canvas of target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Calculate padding and center the image
            dx = (target_w - new_w) // 2
            dy = (target_h - new_h) // 2
            canvas[dy:dy+new_h, dx:dx+new_w] = resized
            
            return canvas
        except:
            return None