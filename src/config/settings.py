import os

class Settings:
    """Application settings and configuration."""
    
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "../model")
        self.persons_dir = os.path.join(self.base_dir, "../persons")
        
        # YOLO model paths
        self.yolo_weights = os.path.join(self.models_dir, "yolov4.weights")
        self.yolo_cfg = os.path.join(self.models_dir, "yolov4.cfg")
        self.yolo_names = os.path.join(self.models_dir, "coco.names")
        
        # Camera settings
        self.camera_window_name = "Person Recognition"
        self.camera_timeout = 3.0
        
        # Person recognition settings
        self.person_image_min_count = 10
        self.feature_image_width = 128
        self.feature_image_height = 256
        self.target_capture_size = (128*4, 256*4)
        
        # Detection settings
        self.detection_confidence = 0.75