import cv2
import numpy as np
from unitree_sdk2py.go2.video.video_client import VideoClient

class VideoHandler:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 1600)

        # Initialize VideoClient
        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()
    
    def get_image(self):
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

    def display_image(self, image, distance = 0, angle = 0):
        try:
            if image is None or image.size == 0:
                print("Warning: Cannot display empty/None image")
                return
            
            # Validate image dimensions
            height, width = image.shape[:2]
            if width <= 0 or height <= 0:
                print(f"Warning: Invalid image dimensions {width}x{height}")
                return

            # Convert distance and angle to strings
            distance_text = f"Distance: {distance:.2f} m"
            angle_text = f"Angle: {angle:.2f} deg"
            
            # Get image dimensions
            height, width, _ = image.shape
            
            # Set text position (middle-lower part of the image)
            text_x = width // 2 - 100  # Centered horizontally
            text_y = height - 30      # 30 pixels from the bottom
            
            # Set font parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 255, 0)  # Green color
            thickness = 2
            
            # Put distance text
            cv2.putText(image, distance_text, (text_x, text_y - 30), 
                    font, font_scale, font_color, thickness)
            
            # Put angle text
            cv2.putText(image, angle_text, (text_x, text_y), 
                    font, font_scale, font_color, thickness)
            

            cv2.imshow(self.window_name, image)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            