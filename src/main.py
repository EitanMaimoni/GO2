import sys
import cv2
import os
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from appearance_model_manager import AppearanceModelManager
from video_handler import VideoHandler
from detection import PersonDetector
from robot_movement import RobotMovement

def main():
    # Initialize DDS communication
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # Paths to YOLO files
    weights_path = "../model/yolov4.weights"
    cfg_path = "../model/yolov4.cfg"
    names_path = "../model/coco.names"

    # Initialize video handler
    window_name = "front_camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1600)
    video_handler = VideoHandler(window_name)

    # Initialize person detector
    detector = PersonDetector(weights_path, cfg_path, names_path)
    
    # Initialize person model manager
    model_manager = AppearanceModelManager()

    # Initialize robot movement
    robot_movement = RobotMovement()

    try:
        print("\nOptions:")
        print("1. Create new person model (appearance-based)")
        print("2. Use existing person model (appearance-based)")
        print("3. Just detect people (no specific person recognition)")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            # Create new person model
            person_name = input("Enter person name: ").strip()
            print(f"Creating dataset for {person_name}. Press ENTER to capture image, ESC when done.")
            
            model_manager = AppearanceModelManager()
            capture_window = "Capturing Person - Press ENTER to save, ESC to finish"
            cv2.namedWindow(capture_window, cv2.WINDOW_NORMAL)
            
            try:
                while True:
                    image = video_handler.get_image()
                    if image is None:
                        print("Received bad image, ignoring...")
                        continue
                    
                    # Get cropped person images
                    images = detector.get_cropped_persons(image)
                    if not images:
                        # Show full image if no person detected
                        cv2.imshow(capture_window, image)
                        print("No person detected in frame")
                    else:
                        # Show the cropped person
                        cv2.imshow(capture_window, images[0])
                    
                    # Check for key press - wait for 1ms and check
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 13:  # ENTER key
                        if images:
                            try:
                                model_manager.save_person_image(person_name, images[0])
                                print(f"Saved image {len(os.listdir(os.path.join(model_manager.base_dir, person_name, 'raw')))}")
                            except Exception as e:
                                print(f"Error saving image: {e}")
                        else:
                            print("No person detected to save!")
                    elif key == 27:  # ESC key
                        print("Finished capturing images")
                        break
                    elif key != 255:  # 255 means no key pressed
                        print(f"Debug: Key pressed with code {key}")
            
            finally:
                cv2.destroyWindow(capture_window)
            
            # After collecting images, train the model
            try:
                num_images = len(os.listdir(os.path.join(model_manager.base_dir, person_name, 'raw')))
                if num_images >= 10:
                    print(f"Found {num_images} images. Now training model...")
                    model_path = model_manager.train_person_model(person_name)
                    print(f"Model trained and saved to {model_path}")
                else:
                    print(f"Need at least 10 images, only got {num_images}. Model not trained.")
            except Exception as e:
                print(f"Error in model training: {e}")

        elif choice == "2":
            # Use existing model
            model_manager = AppearanceModelManager()
            models = model_manager.list_existing_models()
            if not models:
                print("No existing models found!")
                return
                
            print("Available models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")
            
            model_choice = int(input("Select model (number): ").strip())
            selected_model = models[model_choice-1]
            
            # Load the model
            try:
                model, pca = model_manager.load_person_model(selected_model)
                print(f"Loaded model for {selected_model}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
            
            # Main detection loop
            while True:
                image = video_handler.get_image()
                if image is not None:
                    # Detect specific person
                    detected, confidence, person_img = detector.detect_specific_person_appearance(
                    image, model, pca)

                if detected:
    
                    cv2.imshow(window_name, person_img)
                    
                if cv2.waitKey(20) == 27:
                    robot_movement.stop()
                    break

        # elif choice == "3":
        #     # Original detection behavior
        #     while True:
        #         image = video_handler.get_image()
        #         if image is not None:
        #             person_detected, distance, angle, image = detector.detect_person(image)
        #             video_handler.display_image(image, distance, angle)
                    
        #             if person_detected:
        #                 robot_movement.walk_towards_target(angle, distance)
                
        #         if cv2.waitKey(20) == 27:
        #             robot_movement.stop()
        #             break
    finally:
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()