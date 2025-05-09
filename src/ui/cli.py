import cv2
import numpy as np
import time

class CLIInterface:
    def __init__(self, system):
        self.system = system
        self.capture_mode = False
        self.latest_image = None
        self.latest_person = None

    # TODO: ADD option to delete model
    def start(self):
        while True:
            print("\n===== Person Tracker CLI =====")
            print("1. Create Person Model")
            print("2. Follow Person")
            print("3. Exit")
            choice = input("Select an option: ").strip()

            if choice == "1":
                name = input("Enter person name: ").strip()
                self.create_model(name)
            elif choice == "2":
                self.follow_person()
            elif choice == "3":
                self.system.cleanup()
                print("Goodbye.")
                break
            else:
                print("Invalid choice.")

    #TODO: Extract the logic of creating model into model_manager class (or even new class that model manager will use)
    def create_model(self, name):
        self.system.model_manager.create_dataset(name)
        self.capture_mode = True
        self.capture_count = 0

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        while self.capture_mode:
            frame = self.system.camera.get_frame()
            if frame is None:
                continue

            person_img, _ = self.system.detector.get_first_person(frame)
            self.latest_image = frame
            self.latest_person = person_img

            display_img = person_img if person_img is not None else frame
            cv2.imshow("Tracking", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if self.latest_person is not None:
                    self.system.model_manager.save_image(name, self.latest_person)
                    self.capture_count += 1
                    print(f"Saved image {self.capture_count}")
            elif key == 27:  # ESC
                self.capture_mode = False
                print("Training model...")
                try:
                    self.system.model_manager.train_model_osnet(name)
                    print(f"Model trained successfully for {name}.")
                except Exception as e:
                    print(f"Training failed: {e}")
                cv2.destroyWindow("Tracking")

    # TODO: extract the logic of tracking into a separate class
    def follow_person(self):
        models = self.system.model_manager.list_models()
        if not models:
            print("No trained models available.")
            return

        print("Available models:")
        for i, name in enumerate(models):
            print(f"{i + 1}. {name}")

        idx = int(input("Select person to follow: ")) - 1
        name = models[idx]

        gallery_features = np.load(f"{self.system.model_manager.base_dir}/{name}/gallery/features.npy")

        self.capture_mode = True
        print("Tracking started. Press ESC to stop.")
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

        last_seen_time = time.time()

        while self.capture_mode:
            frame = self.system.camera.get_frame()
            if frame is None:
                continue

            person_img, target_info = self.system.recognizer.recognize_target(frame, gallery_features)

            if person_img is not None:
                if target_info is not None:
                    annotated = self.system.visualizer.draw_tracking_info(
                        person_img,
                        distance=target_info.get("distance", 0),
                        angle=target_info.get("angle", 0),
                        confidence=target_info.get("confidence", 0)
                    )
                    last_seen_time = time.time()
                    display_img = annotated
                    self.system.robot.follow_target(target_info["angle"], target_info["distance"])
                else:
                    display_img = person_img
            else:
                display_img = frame
                self.system.robot.follow_target(0, 0)

            time_since_seen = time.time() - last_seen_time
            if time_since_seen >= 3.0:
                self.system.robot.follow_target(0, 0)

            cv2.imshow("Tracking", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.capture_mode = False
                self.system.robot.follow_target(0, 0)
                print("Stopped tracking.")
                cv2.destroyWindow("Tracking")
