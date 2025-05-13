import cv2
import numpy as np
import math
import time
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp


class PersonRecognition:
    """Person recognition class: detect -> extract feature -> match to gallery."""

    def __init__(self, detector, feature_extractor, settings):
        """
        Initialize person tracker.
        
        Args:
            detector: PersonDetector instance
            feature_extractor: FeatureExtractor instance
            regocnition_confidence: Threshold for cosine similarity
        """
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.regocnition_confidence = settings.regocnition_confidence
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def _remove_background(self, image):
        """Remove background using MediaPipe SelfieSegmentation."""
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_selfie_segmentation.process(rgb_img)

        if not results.segmentation_mask.any():
            return image  # fallback

        mask = results.segmentation_mask > 0.1
        bg_removed = np.where(mask[..., None], image, (0, 0, 0)).astype(np.uint8)
        return bg_removed


    def recognize_target(self, frame, gallery_features):
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

        # TODO: Compare all confidence scores (between each person detected) and select the best one (for getting the best match)
        # another option is to take the first one detected and not checking the other (for getting the fastest response)
        detections = self.detector.detect_persons(frame)
        print(f"[Detection] Found {len(detections)} persons")
        for detection in detections:
            x, y, w, h = detection['box']
            person_img = frame[y:y+h, x:x+w]
            if person_img.size == 0:
                continue

            # Extract feature and compare
            # TODO: Maybe its better to resize to target size (force exact dimensions, the model trained on this)
            feature = self.feature_extractor.extract(self._remove_background(person_img))
            if feature is None:
                continue
            
            similarities = cosine_similarity(feature, gallery_features)[0]
            top_k = min(50, len(similarities))
            top_k_similarities = np.sort(similarities)[-top_k:]
            confidence = np.mean(top_k_similarities)
            is_target = confidence >= self.regocnition_confidence

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

    

