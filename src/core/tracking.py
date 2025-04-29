import cv2
import numpy as np
import math
import time
from sklearn.metrics.pairwise import cosine_similarity

class PersonTracker:
    """Simplified tracker: detect -> extract feature -> match to gallery in runtime."""

    def __init__(self, detector, feature_extractor, similarity_threshold=0.8):
        """
        Initialize person tracker.
        
        Args:
            detector: PersonDetector instance
            feature_extractor: FeatureExtractor instance
            similarity_threshold: Threshold for cosine similarity
        """
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold

    def track_target(self, frame, gallery_features):
        """
        Track the target person in the frame by comparing with gallery features.
        
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
        highest_confidence = 0

        height, width = frame.shape[:2]

        detections = self.detector.detect_persons(frame)
        print(f"[Detection] Found {len(detections)} persons")
        for detection in detections:
            x, y, w, h = detection['box']
            person_img = frame[y:y+h, x:x+w]
            if person_img.size == 0:
                continue

            # Extract feature and compare
            feature = self.feature_extractor.extract(person_img)
            if feature is None:
                continue
            
            start = time.perf_counter()

            similarities = cosine_similarity(feature, gallery_features)[0]
            top_k = min(50, len(similarities))
            top_k_similarities = np.sort(similarities)[-top_k:]
            confidence = np.mean(top_k_similarities)
            is_target = confidence >= self.similarity_threshold

            elapsed = time.perf_counter() - start
            print(f"[Timing] Similarity calc took {elapsed:.6f}s")

            color = (0, 255, 0) if is_target else (0, 0, 255)
            cv2.rectangle(visualized_frame, (x, y), (x+w, y+h), color, 2)

            if is_target:
                label = f"Target: {confidence:.2f}"
                cv2.putText(visualized_frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if confidence > highest_confidence:
                    highest_confidence = confidence

                    x_center = x + w / 2
                    y_bottom = y + h
                    x_adjusted = x_center - (width / 2)
                    y_adjusted = height - y_bottom

                    target_detection = {
                        'box': (x, y, w, h),
                        'confidence': confidence,
                        'distance': self._estimate_distance_from_bottom(y_adjusted),
                        'angle': self._estimate_angle(x + w/2, width)
                    }
            else:
                label = f"Person: {detection.get('confidence', 0):.2f}"
                cv2.putText(visualized_frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return visualized_frame, target_detection

    def _estimate_distance_from_bottom(self, y_from_bottom):
        return y_from_bottom / 100

    def _estimate_angle(self, center_x, img_width):
        # 70 is the pove of camera, need to do it threw settings
        return ((center_x - (img_width / 2)) / (img_width / 2)) * (70.0 / 2)

