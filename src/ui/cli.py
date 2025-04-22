import cv2
import os

class CommandLineInterface:
    """Command line interface for the person following system."""

    def __init__(self, system):
        """
        Initialize CLI.

        Args:
            system: PersonFollowingSystem instance
        """
        self.system = system

    def start(self):
        """
        Start CLI and present options to user.
        """
        try:
            while True:
                print("\nPerson Following System")
                print("======================")
                print("Options:")
                print("1. Create new person model")
                print("2. Follow specific person")
                choice = input("Enter your choice (1-2): ").strip()
                
                if choice == "1":
                    self.create_model_workflow()
                elif choice == "2":
                    self.follow_person_workflow()
                else:
                    print("Invalid choice")
        except KeyboardInterrupt:
            print("\nProgram terminated by user")
        finally:
            self.system.cleanup()

    def create_model_workflow(self):
        """
        Handle the workflow for creating a new person model.
        """
        person_name = input("Enter person name: ").strip()
        if not person_name:
            print("Person name cannot be empty")
            return

        print(f"Creating dataset for {person_name}. Press ENTER to capture image, ESC when done.")

        # Create directory structure for new person
        self.system.model_manager.create_dataset(person_name)

        capture_window = "Capturing Person - Press ENTER to save, ESC to finish"
        cv2.namedWindow(capture_window, cv2.WINDOW_NORMAL)

        image_count = 0

        try:
            while True:
                frame = self.system.camera.get_frame()
                if frame is None:
                    continue

                # Get cropped person images (using first person found)
                person_img, _ = self.system.detector.get_first_person(frame)

                if person_img is None:
                    cv2.imshow(capture_window, frame)
                    print("No person detected in frame")
                else:
                    cv2.imshow(capture_window, person_img)

                key = cv2.waitKey(1) & 0xFF

                if key == 13:  # ENTER key
                    if person_img is not None:
                        self.system.model_manager.save_image(person_name, person_img)
                        image_count += 1
                        print(f"Saved image {image_count}")
                elif key == 27:  # ESC key
                    break

            # Train the model with collected images
            try:
                self.system.model_manager.train_model(person_name)
                print(f"Model trained successfully for {person_name}")
            except Exception as e:
                print(f"Error in model training: {e}")

        finally:
            cv2.destroyWindow(capture_window)

    def follow_person_workflow(self):
        """
        Handle the workflow for following a specific person.
        """
        models = self.system.model_manager.list_models()
        if not models:
            print("No trained models available")
            return

        print("Available persons:")
        for i, name in enumerate(models, 1):
            print(f"{i}. {name}")

        try:
            choice = int(input("Select person to follow (number): ")) - 1
            person_name = models[choice]
        except (ValueError, IndexError):
            print("Invalid selection")
            return

        if not self.system.recognizer.load_target(person_name):
            print("Failed to load target model")
            return

        print(f"Following {person_name} (Green=You, Red=Others). Press ESC to stop.")

        try:
            while True:
                frame = self.system.camera.get_frame()
                if frame is None:
                    continue

                visualized_frame, target_info = self.system.tracker.track_target(frame)
                annotated_frame = self.system.visualizer.draw_tracking_info(
                    visualized_frame,
                    distance=target_info['distance'] if target_info else 0,
                    angle=target_info['angle'] if target_info else 0
                )

                self.system.camera.display_image(annotated_frame)

                if target_info:
                    self.system.robot.follow_target(target_info['angle'], target_info['distance'])

                if cv2.waitKey(1) == 27:  # ESC key
                    self.system.robot.stop()
                    break

        finally:
            self.system.cleanup()