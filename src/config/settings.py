import os

class Settings:
    """Application settings and configuration."""
    
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "../model")
        self.persons_dir = os.path.join(self.base_dir, "../persons")
        
        # YOLO model paths
        self.yolo_weights = os.path.join(self.models_dir, "yolov4-tiny.weights")
        self.yolo_cfg = os.path.join(self.models_dir, "yolov4-tiny.cfg")
        self.yolo_names = os.path.join(self.models_dir, "coco.names")
        
        # Camera settings
        self.camera_timeout = 3.0
        
        # Person recognition settings
        self.person_image_min_count = 100
        
        # Detection settings
        self.detection_confidence = 0.75

        # Tracking settings
        self.regocnition_confidence = 0.75
