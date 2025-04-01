import cv2
import numpy as np

class SpecificPersonFollower:
    """
    A class that handles tracking and following a specific person with visual feedback.
    
    Responsibilities:
    1. Identify target person among all detected persons
    2. Provide visual feedback with color-coded bounding boxes:
       - Green: Target person (you)
       - Red: Other persons
    3. Calculate and execute robot movement commands
    
    Works with:
    - PersonDetector: For person detection
    - AppearanceModelManager: For person recognition
    - RobotMovement: For physical movement
    """
    
    def __init__(self, detector, model_manager):
        """
        Initialize the person follower with required components.
        
        Args:
            detector (PersonDetector): Person detection instance
            model_manager (AppearanceModelManager): Model management instance
            robot (RobotMovement): Robot control instance
        """
        self.detector = detector
        self.model_manager = model_manager
        self.target_model = None
        self.target_pca = None
        
    def set_target_person(self, person_name):
        """
        Load the recognition model for the target person.
        
        Args:
            person_name (str): Name of registered person to follow
            
        Raises:
            FileNotFoundError: If no model exists for specified person
        """
        self.target_model, self.target_pca = self.model_manager.load_person_model(person_name)
        
    def process_frame(self, image):
        """
        Process an image frame to identify and visualize persons.
        
        Args:
            image (numpy.ndarray): Input image frame
            
        Returns:
            tuple: (processed_image, target_detection)
            - processed_image: Input image with color-coded bounding boxes
            - target_detection: Dictionary with target person's detection info or None
        """
        all_detections = self.detector.detect_persons(image)
        processed_image = image.copy()
        target_detection = None
        
        
        for detection in all_detections:
            x, y, w, h = detection['box']
            person_img = image[y:y+h, x:x+w]
            
            if person_img.size == 0:
                continue
                
            is_target = self._is_target_person(person_img)
            self._draw_detection(processed_image, detection, is_target)
            
            if is_target:
                target_detection = detection
                
        return processed_image, target_detection
        
    def _is_target_person(self, person_img):
        """
        Check if image matches the target person.
        
        Args:
            person_img (numpy.ndarray): Cropped person image
            
        Returns:
            bool: True if matches target, False otherwise
        """
        features = self.model_manager.extract_appearance_features(person_img)
        reduced_features = self.target_pca.transform([features])
        return self.target_model.predict(reduced_features) == 1
        
    def _draw_detection(self, image, detection, is_target):
        """
        Draw color-coded bounding box for a detection.
        
        Args:
            image (numpy.ndarray): Image to draw on
            detection (dict): Detection info dictionary
            is_target (bool): Whether this is the target person
        """
        x, y, w, h = detection['box']
        color = (0, 255, 0) if is_target else (0, 0, 255)  # BGR colors
        thickness = 2 if is_target else 1
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        
        label = "TARGET" if is_target else "Person"
        cv2.putText(image, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
        