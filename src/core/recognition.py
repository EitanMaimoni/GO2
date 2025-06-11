import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

class PersonRecognition:
    """Person recognition class: detect -> extract feature -> match to gallery."""

    def __init__(self, feature_extractor, settings):
        """
        Initialize person tracker.
        
        Args:
            feature_extractor: FeatureExtractor instance
            settings: Settings instance
        """
        self.feature_extractor = feature_extractor
        self.regocnition_confidence = settings.regocnition_confidence
    
    def recognize_target(self, frame, detections, gallery_features):
        """
        Recognize target in the frame.  

        Args:
            frame: The camera frame
            gallery_features: The gallery features of the person to match against
        
        Returns:
            Tuple of (visualized_frame, target_info)
        """
        if frame is None:
            return None, None

        visualized_frame = frame.copy()
        target_detection = None

        # TODO: Use threads to compute scores in parallel, pick the best one.
        for detection in detections:
            x, y, w, h = detection['box']
            person_img = frame[y:y+h, x:x+w]
            if person_img.size == 0:
                continue

            # Extract feature and compare
            feature = self.feature_extractor.extract(person_img)
            if feature is None:
                continue
            
            # Use cosine similarity to compare direction (pattern) of feature vectors, ignoring magnitude differences caused by lighting or scale
            similarities = cosine_similarity(feature, gallery_features)[0]

            # Use max similarity (like in test code)
            confidence = np.max(similarities)

            is_target = (confidence >= self.regocnition_confidence)

            color = (0, 255, 0) if is_target else (0, 0, 255)
            cv2.rectangle(visualized_frame, (x, y), (x+w, y+h), color, 2)

            if is_target:
                label = f"Target: {confidence:.2f}"
                cv2.putText(visualized_frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                target_detection = {
                    'box': (x, y, w, h),
                    'confidence': confidence,
                    'distance': detection.get('distance', 0),
                    'angle': detection.get('angle', 0)
                }
            else:
                label = f"Person: {confidence:.2f}"
                cv2.putText(visualized_frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return visualized_frame, target_detection

    

