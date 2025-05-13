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
        # Load the YOLO model:
        # - cfg defines the network structure (e.g., number of layers, types of layers)
        # - weights contains the trained values
        # This create OpenCV's DNN module based on the config file and then initializes it with the weights.
        self.net = cv2.dnn.readNet(settings.yolo_weights, settings.yolo_cfg)

        # YOLO model itself only returns class IDs (e.g., 0, 1, 2...)
        # The yolo_names file contains the mapping of these IDs to human-readable class names ((line) 1 - person, (line) 2 - bicycle, etc.)
        # We read the file and store the class names in a list.
        with open(settings.yolo_names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get the names of the output layers used by YOLO (e.g., ['conv_28', ..., 'yolo_30', ..., 'yolo_37'])
        layer_names = self.net.getLayerNames() 

        # Each yolo_x layer is a detection layer. we need to get the indices of these layers.
        unconnected_out_layers = self.net.getUnconnectedOutLayers()

        # Convert indices to layer names (forward() will use these names)
        # - Some versions return 2D array ([[200], [267]])
        # - Others return 1D array ([200, 267])
        # Subtract 1 because OpenCV uses 1-based indexing, but Python lists use 0-based
        if unconnected_out_layers.ndim == 2:
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
        else:
            self.output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

        # Get the input size of the YOLO model from the cfg file
        self.width_intput, self.height_intput = self.get_input_size_from_cfg(settings.yolo_cfg)

        # Set the confidence threshold for detection
        self.confidence_threshold = settings.detection_confidence

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

        # Convert input image to a 4D blob:
        # Output blob has shape: [1, 3, x, x] → ready for network input
        blob = cv2.dnn.blobFromImage(
            image, 
            1/255.0,         # - Scales pixel values to [0,1] by multiplying by 1/255
            (self.width_intput, self.height_intput),         # - Resizes to (x, x) as our model expect
            (0, 0, 0),       # - Does not subtract any mean (0,0,0)
            True,            # - Converts BGR to RGB (swapRB=True)
            crop=False       # - Does not crop the image (crop=False)
        )

        # Feed the image blob into the network (set the input for the network)
        self.net.setInput(blob)

        # Run a forward pass through the network
        # - self.output_layers contains the names of YOLO's final detection layers (e.g., ["yolo_82", "yolo_94", "yolo_106"])
        # - Each of these layers is responsible for detecting objects at a different scale (large, medium, small)
        # - YOLO uses feature maps of different sizes to detect objects of different sizes:
        #     - e.g., one layer might divide the image into a 13×13 grid (for large objects)
        #     - another might use a 26×26 grid (medium objects)
        #     - another a 52×52 grid (small objects)
        # - At each grid cell, YOLO applies several predefined anchor boxes (usually 3) to guess object shapes
        # - For each grid cell and each anchor box, YOLO outputs one detection candidate (i.e., one row)
        # - Therefore, each output layer produces a large number of rows = (grid_size × grid_size × anchors)
        # - outs is a list of NumPy arrays — one per detection layer
        #     - Each array has shape (num_detections, 85), where each row contains:
        #         [ center_x, center_y, width, height, objectness, class_0, class_1, ..., class_79 ]
        #         - center_x, center_y, width, height → bounding box (relative to image size)
        #         - objectness → confidence there's *any* object in this box
        #         - class_i → confidence that it's a specific class (e.g., person, car, dog, etc.)
        #     - Most rows are low-confidence guesses or overlapping duplicates
        # - We will later filter these rows using a confidence threshold and Non-Maximum Suppression (NMS)
        outs = self.net.forward(self.output_layers) 
        
        boxes = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold and self.classes[class_id] == 'person':
                    # Save the bounding box coordinates in NMS and OpenCV format and the confidence score
                    # YOLO outputs are relative values in [0, 1]
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Validate the bounding box coordinates
                    x, y, w, h = self._validate_bbox(x, y, w, h, width, height)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
                    boxes,              # List of all (x, y, w, h) boxes we found
                    confidences,        # Their corresponding confidence scores
                    self.confidence_threshold,  # Minimum confidence to keep a box
                    0.4                 # IOU threshold: if two boxes overlap > 40%, drop the lower one
                )

        # Choose the detections that passed NMS
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            x, y, w, h = boxes[i]
        
            y_adjusted = height - (y + h)
            distance = self._estimate_distance_from_bottom(y_adjusted)
            angle = self._estimate_angle(x + w / 2, width)

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
        
        return person_img, detection

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
    
    def _estimate_distance_from_bottom(self, y_from_bottom):
        return y_from_bottom / 100

    def _estimate_angle(self, center_x, img_width):
        # 70 is the pove of camera, need to do it threw settings
        return ((center_x - (img_width / 2)) / (img_width / 2)) * (70.0 / 2)
    
    def get_input_size_from_cfg(self, cfg_path):
        width = 416  # default fallback
        height = 416

        with open(cfg_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("width="):
                    width = int(line.split("=")[1])
                elif line.startswith("height="):
                    height = int(line.split("=")[1])
                if width and height:
                    break

        return width, height
