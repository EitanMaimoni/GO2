class RobotParams:
    """Robot-specific parameters."""
    
    def __init__(self):
        # Movement parameters
        self.angle_margin = 7.0  # Degrees
        self.distance_threshold = 3.5  # Meters
        self.rotation_speed = 0.35
        self.forward_speed_factor = 0.35
        self.combined_speed_factor = 0.5
        
        # Camera parameters
        self.camera_fov = 70.0  # Field of view in degrees
        self.known_person_height = 1.7  # Average person height in meters
        self.focal_length = 600  # Camera focal length in pixels