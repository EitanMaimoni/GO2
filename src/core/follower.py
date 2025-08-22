import cv2
import time
import numpy as np

class PersonFollower:

    def __init__(self, robot, recognizer, visualizer, camera, detector, model_manager):
        """
        Initialize person follower.
        
        Args:
            robot: RobotController instance
            recognizer: PersonRecognition instance
            visualizer: Visualizer instance
            settings: Settings instance
        """
        self.robot = robot
        self.recognizer = recognizer
        self.visualizer = visualizer
        self.camera = camera
        self.detector = detector
        self.model_manager = model_manager
        

    def follow_person(self):
        models = self.model_manager.list_models()
        if not models:
            print("No trained models available.")
            return

        print("Available models:")
        for i, name in enumerate(models):
            print(f"{i + 1}. {name}")

        idx = int(input("Select person to follow: ")) - 1
        name = models[idx]

        gallery_features = np.load(f"{self.model_manager.base_dir}/{name}/gallery/features.npy")

        print("Tracking started. Press ESC to stop.")
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

        last_seen_time = time.time()

        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            detections = self.detector.detect_persons(frame)
            print(f"[Detection] Found {len(detections)} persons")
            person_img, target_info = self.recognizer.recognize_target(frame, detections, gallery_features)

            if person_img is not None:
                if target_info is not None:
                    annotated = self.visualizer.draw_tracking_info(
                        person_img,
                        distance=target_info.get("distance", 0),
                        angle=target_info.get("angle", 0),
                        confidence=target_info.get("confidence", 0)
                    )
                    last_seen_time = time.time()
                    display_img = annotated
                    self.robot.follow_target(target_info["angle"], target_info["distance"])
                else:
                    display_img = person_img
            else:
                display_img = frame

            time_since_seen = time.time() - last_seen_time
            if time_since_seen >= 1.0:
                self.robot.follow_target(0, 0)

            cv2.imshow("Tracking", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.robot.follow_target(0, 0)
                print("Stopped tracking.")
                cv2.destroyWindow("Tracking")
                break