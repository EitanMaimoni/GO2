import cv2

class PersonTracker:
    """Tracks detected persons across video frames."""
    
    def __init__(self, detector, recognizer):
        """
        Initialize person tracker.
        
        Args:
            detector: PersonDetector instance
            recognizer: PersonRecognizer instance
        """
        self.detector = detector
        self.recognizer = recognizer
    
    def track_target(self, frame):
        """
        Track target person in frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (visualized_frame, target_info)
                - visualized_frame: Frame with visualization
                - target_info: Dictionary with target info or None if not found
        """
        if frame is None:
            return None, None
            
        # Get all person detections
        all_detections = self.detector.detect_persons(frame)
        visualized_frame = frame.copy()
        target_detection = None
        
        # Find target among all detections
        for detection in all_detections:
            x, y, w, h = detection['box']
            person_img = frame[y:y+h, x:x+w]
            
            if person_img.size == 0:
                continue
                
            is_target = self.recognizer.is_target_person(person_img)
            
            # Draw box
            color = (0, 255, 0) if is_target else (0, 0, 255)  # BGR: Green for target, Red for others
            thickness = 2 if is_target else 1
            cv2.rectangle(visualized_frame, (x, y), (x+w, y+h), color, thickness)
            
            # Draw label
            label = "TARGET" if is_target else "Person"
            cv2.putText(visualized_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if is_target:
                target_detection = detection
        
        return visualized_frame, target_detection