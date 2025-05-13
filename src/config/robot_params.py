class RobotParams:
    """Robot-specific parameters."""
    
    def __init__(self):
        # Movement parameters
        self.angle_threshold = 7.0 # Angle threshold (Dont rotate if angle is less than this)
        self.distance_threshold = 3.5  # Distance threshold (Dont move if distance is less than this)
        self.forward_speed_factor = 2  # Speed factor for forward movement
        self.rotation_speed = 0.6 # Rotation speed in radians per second
        
        # Camera parameters
        self.camera_fov = 70.0  # Field of view in degrees
