import sys
import cv2
import os
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from person_detector import SpecificPersonFollower
from appearance_model_manager import AppearanceModelManager
from detection import PersonDetector
from video_handler import VideoHandler
from robot_movement import RobotMovement

def main():
    """
    Main entry point for the person recognition and following system.
    
    Provides two main functionalities:
    1. Create new appearance models for persons
    2. Follow a specific person (to be implemented)
    
    Usage:
        python main.py [DDS_config_file]
    """
    # Initialize DDS communication
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # Paths to YOLO files
    weights_path = "../model/yolov4.weights"
    cfg_path = "../model/yolov4.cfg"
    names_path = "../model/coco.names"

    # Initialize components
    video_handler = VideoHandler("Person Recognition")
    detector = PersonDetector(weights_path, cfg_path, names_path)
    model_manager = AppearanceModelManager()
    robot = RobotMovement()

    try:
        print("\nOptions:")
        print("1. Create new person model")
        print("2. Follow specific person")
        choice = input("Enter your choice (1-2): ").strip()

        if choice == "1":
            create_new_model(model_manager, video_handler, detector)
        elif choice == "2":
            follow_person(model_manager, video_handler, detector, robot)
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    finally:
        cv2.destroyAllWindows()

def create_new_model(model_manager, video_handler, detector):
    """
    Handle the creation of a new person appearance model.
    
    Args:
        model_manager: AppearanceModelManager instance
        video_handler: VideoHandler instance
        detector: PersonDetector instance
        
    Flow:
    1. Prompts for person name
    2. Captures images when ENTER is pressed
    3. Saves images to person's dataset
    4. Trains model when ESC is pressed
    """
    person_name = input("Enter person name: ").strip()
    print(f"Creating dataset for {person_name}. Press ENTER to capture image, ESC when done.")
    
    # Create directory structure for new person
    model_manager.create_new_person_dataset(person_name)
    
    capture_window = "Capturing Person - Press ENTER to save, ESC to finish"
    cv2.namedWindow(capture_window, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            image = video_handler.get_image()
            if image is None:
                continue
            
            # Get cropped person images (using first person found)
            person_img = detector.get_first_person(image)
            
            if person_img is None:
                cv2.imshow(capture_window, image)
                print("No person detected in frame")
            else:
                cv2.imshow(capture_window, person_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER key
                if person_img is not None:
                    model_manager.save_person_image(person_name, person_img)
                    print(f"Saved image {len(os.listdir(os.path.join(model_manager.base_dir, person_name, 'raw')))}")
            elif key == 27:  # ESC key
                break
        
        # Train the model with collected images
        try:
            model_manager.train_person_model(person_name)
            print(f"Model trained successfully for {person_name}")
        except Exception as e:
            print(f"Error in model training: {e}")
            
    finally:
        cv2.destroyWindow(capture_window)

def follow_person(model_manager, video_handler, detector, robot):
    """
    Main person following routine with visual feedback.
    
    Flow:
    1. Lists available person models
    2. Lets user select target person
    3. Initializes follower and processes video stream
    4. Provides real-time visual feedback:
       - Green boxes: Target person
       - Red boxes: Other persons
    
    Args:
        model_manager (AppearanceModelManager): Model management instance
        video_handler (VideoHandler): Video input handler
        detector (PersonDetector): Person detection instance
        robot (RobotMovement): Robot control instance
    """
    models = model_manager.list_existing_models()
    if not models:
        print("No trained models available")
        return
    
    print("Available persons:")
    for i, name in enumerate(models, 1):
        print(f"{i}. {name}")
    
    choice = int(input("Select person to follow (number): ")) - 1
    person_name = models[choice]
    
    follower = SpecificPersonFollower(detector, model_manager)
    follower.set_target_person(person_name)
    
    print(f"Following {person_name} (Green=You, Red=Others). ESC to stop.")
    
    try:
        while True:
            frame = video_handler.get_image()
            if frame is None:
                continue
                
            visualized_frame, person_detected = follower.process_frame(frame)
            

            # Uncomment the following lines after finishing deb
            if person_detected:
                video_handler.display_image(visualized_frame, person_detected['angle'], person_detected['distance'])
                # Walk towards the person
                # robot.walk_towards_target(person_detected['angle'], person_detected['distance'])
            else:
                video_handler.display_image(visualized_frame, 0, 0)
                # Stop robot movement if no person detected
                # robot.walk_towards_target(0, 0)

            if cv2.waitKey(1) == 27:  # ESC key
                break
                
    finally:
        cv2.destroyAllWindows()
        robot.stop()

if __name__ == "__main__":
    main()