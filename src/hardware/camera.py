import cv2
import numpy as np
from unitree_sdk2py.go2.video.video_client import VideoClient

class Camera:
    """Interface to the robot's camera."""
    
    def __init__(self, settings):
        """
        Initialize camera.
        
        Args:
            settings: Application settings
        """
        # Initialize VideoClient
        self.client = VideoClient()
        self.client.SetTimeout(settings.camera_timeout)
        self.client.Init()
    
    def get_frame(self):
        """
        Get current camera frame.
        
        Returns:
            numpy.ndarray: Camera frame or None on error
        """
        try:
            code, data = self.client.GetImageSample()
            if code == 0:
                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if image is not None and image.size > 0:  
                    return image
                
            print("Get image sample error. code:", code)
            return None
        
        except Exception as e:
            print(f"Error getting image: {str(e)}")
            return None
    
    def display_image(self, image):
        """
        Display image in window.
        
        Args:
            image: Image to display
        """
        if image is None or image.size == 0:
            print("Warning: Cannot display empty/None image")
            return
            
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)
    
    def release(self):
        """Release camera resources."""
        cv2.destroyAllWindows()