import cv2
import numpy as np
from unitree_sdk2py.go2.video.video_client import VideoClient

class VideoHandler:
    def __init__(self, window_name, window):
        self.window_name = window_name
        self.window = window

        # Initialize VideoClient
        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()
    
    def get_image(self):
        code, data = self.client.GetImageSample()
        if code == 0:
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return image
        else:
            print("Get image sample error. code:", code)
            return None

    def display_image(self, image):
        if image is not None:
            cv2.imshow(self.window_name, image)
            cv2.waitKey(1)
        else:
            print("No image to display.")
            
    def cleanup(self):
        # Note: Don't destroy the window here since it's managed by main
        pass