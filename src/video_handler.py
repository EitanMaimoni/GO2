import cv2
import numpy as np
from unitree_sdk2py.go2.video.video_client import VideoClient
from detection import PersonDetector
from robot_movement import RobotMovement

class VideoHandler:
    def __init__(self):
        self.window_name = "front_camera"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 1600)

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
        
        if cv2.waitKey(20) == 27:
            self.cleanup()
            exit(0)
            
    def cleanup(self):
        cv2.destroyWindow(self.window_name)


        