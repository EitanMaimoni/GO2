import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from video_handler import VideoHandler
from detection import PersonDetector
from robot_movement import RobotMovement

if __name__ == "__main__":
    # Initialize DDS communication
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])  # Pass network interface as argument
    else:
        ChannelFactoryInitialize(0)  # Use default network interface

    # Paths to YOLO files
    weights_path = "../model/yolov4.weights"
    cfg_path = "../model/yolov4.cfg"
    names_path = "../model/coco.names"

    # Initialize video handler
    video_handler = VideoHandler()

    # Initialize person detector
    detector = PersonDetector(weights_path, cfg_path, names_path)

    # Initialize robot movement
    robot_movement = RobotMovement()

    while True:
        # Get image from Go2 robot
        image = video_handler.get_image()

        if image is not None:
            # Detect person in the image
            person_detected, distance, angle, image = detector.detect_person(image)
            # Display the image (Original image. added bounding box if person detected)
            video_handler.display_image(image)

            if person_detected:
                # Walk towards the person
                robot_movement.walk_towards_target(angle, distance)
    
        else:
            print("Received bad image, ignoring...")
