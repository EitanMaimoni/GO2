import cv2
import numpy as np
from unitree_sdk2py.go2.video.video_client import VideoClient
from detection import PersonDetector
from robot_movement import RobotMovement

class VideoProcessor:
    def __init__(self, weights_path, cfg_path, names_path):
        self.detector = PersonDetector(weights_path, cfg_path, names_path)
        self.window_name = "front_camera"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 1600)

        # Initialize RobotMovement for controlling the robot
        self.robot_movement = RobotMovement()

    def process_video(self):
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()

        code, data = client.GetImageSample()

        while code == 0:
            try:
                # Get Image data from Go2 robot
                code, data = client.GetImageSample()

                # Convert to numpy image
                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Detect person and calculate distance/angle
                    person_detected, distance, angle, image = self.detector.detect_person(image)

                    # Display the image with distance and angle
                    height, width, _ = image.shape
                    text_x = width // 2 - 100
                    text_y = height - 50

                    # Clear previous text by drawing a filled rectangle
                    cv2.rectangle(image, (text_x - 10, text_y - 30), (text_x + 200, text_y + 10), (0, 0, 0), -1)

                    # Display distance and angle
                    cv2.putText(image, f"Distance: {distance:.2f} m", (text_x, text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(image, f"Angle: {angle:.2f} deg", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display image
                    cv2.imshow(self.window_name, image)

                    # Walk towards the target if a person is detected
                    if person_detected:
                         self.robot_movement.walk_towards_target(angle, distance)
    
                    # Press ESC to stop
                    if cv2.waitKey(20) == 27:
                        break
                else:
                    print("Received bad image, ignoring...")

            except Exception as e:
                print(f"Error processing image: {e}. Ignoring bad image...")
                continue

        if code != 0:
            print("Get image sample error. code:", code)

        cv2.destroyWindow(self.window_name)