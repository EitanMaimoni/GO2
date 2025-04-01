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

    def detect_specific_person_appearance(self, image, model, pca, confidence_threshold=0):
        """Detect if a specific person is in the image based on appearance"""
        cropped_images = self.get_cropped_persons(image)
        for person_img in cropped_images:
            # Extract appearance features
            features = self.extract_appearance_features(person_img)
            
            # Apply PCA transformation
            features_reduced = pca.transform([features])
            
            # Predict with our model
            if hasattr(model, 'predict_proba'):  # For regular SVM
                prediction = model.predict_proba(features_reduced)
                confidence = prediction[0][1]  # Probability it's our person
            else:  # For OneClassSVM
                decision_score = model.decision_function(features_reduced)
                # Convert decision score to a confidence-like value (0-1 range)
                confidence = 1 / (1 + np.exp(-decision_score[0]))  # Sigmoid transformation
            
            if confidence > confidence_threshold:
                return True, float(confidence), person_img  # Explicitly convert to float
        return False, 0.0, None
    
    def extract_appearance_features(self, image):
        """Extract color and texture features (same as in AppearanceModelManager)"""
        # Resize to consistent dimensions
        resized = cv2.resize(image, (128, 256))
        
        # Color features (mean and std of each channel in HSV space)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_mean, h_std = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
        s_mean, s_std = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
        v_mean, v_std = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
        
        # Texture features (using histogram of oriented gradients)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog = self._calculate_hog(gray)
        
        # Combine all features
        features = np.array([
            h_mean, h_std, s_mean, s_std, v_mean, v_std,
            *hog
        ])
        
        return features
    
    def _calculate_hog(self, gray_image, bins=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
        """Calculate Histogram of Oriented Gradients"""
        from skimage.feature import hog
        features = hog(gray_image, orientations=bins, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
        return features