class RobotParams:
    """Robot-specific parameters."""
    
    def __init__(self):
        # Movement parameters
        self.angle_margin = 5.0  # Degrees
        self.distance_threshold = 1.0  # Meters
        self.rotation_speed = 0.5
        self.forward_speed_factor = 0.75
        self.combined_speed_factor = 0.5
        
        # Camera parameters
        self.camera_fov = 70.0  # Field of view in degrees
        self.known_person_height = 1.7  # Average person height in meters
        self.focal_length = 600  # Camera focal length in pixels