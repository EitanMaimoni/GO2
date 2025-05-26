import cv2
import numpy as np
import time
import mediapipe as mp
import os
import glob
import json
import time
from models.model_tester import ModelTester

class CLIInterface:
    def __init__(self, system):
        self.system = system
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


    # TODO: ADD option to delete model
    def start(self):
        while True:
            print("\n===== Person Tracker CLI =====")
            print("1. Create Person Model")
            print("2. Follow Person")
            print("3. Collect training dataset images (TorchReID format)")
            print("4. Test & Fine-tune Model")  # <-- ADD THIS LINE
            print("5. Exit")  # <-- UPDATE number


            choice = input("Select an option: ").strip()

            if choice == "1":
                name = input("Enter person name: ").strip()
                self.create_model(name)
            elif choice == "2":
                self.follow_person()
            elif choice == "3":
                self.collect_finetune_dataset()
            elif choice == "4":  # <-- ADD THIS
                self.test_and_finetune_menu()
            elif choice == "5":  # <-- UPDATE from "4"
                self.system.cleanup()
                print("Goodbye.")
                break
            else:
                print("Invalid choice.")
    
    def _normalize_brightness(self, image, target_mean=128, target_std=50):
        """
        Normalize brightness across images by shifting to a target mean and std.
        """

        if image is None:
            return None
        
        img = image.astype(np.float32)

        current_mean = np.mean(img)
        current_std = np.std(img)

        if current_std < 1e-6:
            return image  # avoid division by near-zero

        # Normalize to zero-mean, unit-std
        img = (img - current_mean) / current_std

        # Scale to target
        img = img * target_std + target_mean
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    #TODO: Extract the logic of creating model into model_manager class (or even new class that model manager will use)
    def create_model(self, name):
        self.system.model_manager.create_dataset(name)
        self.capture_mode = True
        self.capture_count = 0

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        while True:
            frame = self.system.camera.get_frame()
            if frame is None:
                continue

            person_img, _ = self.system.detector.get_first_person(frame)
            person_img = self._normalize_brightness(person_img)

            display_img = person_img if person_img is not None else frame
            cv2.imshow("Tracking", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if person_img is not None:
                    self.system.model_manager.save_image(name, person_img)
                    self.capture_count += 1
                    print(f"Saved image {self.capture_count}")
            elif key == 27:  # ESC
                print("Training model...")
                try:
                    self.system.model_manager.train_model_osnet(name)
                    print(f"Model trained successfully for {name}.")
                except Exception as e:
                    print(f"Training failed: {e}")
                cv2.destroyWindow("Tracking")
                break

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

        # TODO: get full path to the model_manager
        gallery_features = np.load(f"{self.system.model_manager.base_dir}/{name}/gallery/features.npy")

        print("Tracking started. Press ESC to stop.")
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

        last_seen_time = time.time()

        while True:
            frame = self.system.camera.get_frame()
            if frame is None:
                continue

            detections = self.system.detector.detect_persons(frame)
            print(f"[Detection] Found {len(detections)} persons")
            person_img, target_info = self.system.recognizer.recognize_target(frame, detections, gallery_features)

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

            time_since_seen = time.time() - last_seen_time
            if time_since_seen >= 1.0:
                self.system.robot.follow_target(0, 0)

            cv2.imshow("Tracking", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.system.robot.follow_target(0, 0)
                print("Stopped tracking.")
                cv2.destroyWindow("Tracking")
                break
    
    #TODO: Extract the logic of creating model into model_manager class (or even new class that model manager will use)
    def collect_finetune_dataset(self):
        print("\n=== Collect Fine-Tuning Dataset ===")
        print("1. Add person for training")
        print("2. Add person for testing")
        mode = input("Choose mode (1/2): ").strip()

        # Detect next available person_XYZ ID
        def get_next_person_id(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            existing = [d for d in os.listdir(base_dir) if d.startswith("person_")]
            numbers = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
            next_id = max(numbers, default=0) + 1
            return f"person_{next_id:03d}"

        if mode == "1":  # === TRAINING MODE ===
            base_train = "../dataset/train"
            base_val = "../dataset/val"
            person_id = get_next_person_id(base_train)

            train_dir = os.path.join(base_train, person_id)
            val_dir = os.path.join(base_val, person_id)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            train_count = len(glob.glob(os.path.join(train_dir, "img*.jpg")))
            val_count = len(glob.glob(os.path.join(val_dir, "img*.jpg")))
            save_counter = 0

            print(f"[TRAIN MODE] Capturing for {person_id}")
            print("Press ENTER to save image (4 train : 1 val). Press ESC to finish.")
            cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)

            while True:
                frame = self.system.camera.get_frame()
                if frame is None:
                    continue

                person_img, _ = self.system.detector.get_first_person(frame)
                display = person_img if person_img is not None else frame
                cv2.imshow("Capture", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # ENTER
                    if person_img is not None:
                        ext = ".jpg"
                        if save_counter < 4:
                            train_count += 1
                            filename = f"img{train_count:03d}{ext}"
                            dst = os.path.join(train_dir, filename)
                            save_counter += 1
                        else:
                            val_count += 1
                            filename = f"img{val_count:03d}{ext}"
                            dst = os.path.join(val_dir, filename)
                            save_counter = 0
                        cv2.imwrite(dst, person_img)
                        print(f"Saved: {dst}")
                elif key == 27:  # ESC
                    print(f"[DONE] Saved {train_count} train, {val_count} val for {person_id}")
                    cv2.destroyWindow("Capture")
                    break

        elif mode == "2":  # === TESTING MODE ===
            base_query = "../dataset/query"
            base_gallery = "../dataset/gallery"
            person_id = get_next_person_id(base_gallery)

            query_dir = os.path.join(base_query, person_id)
            gallery_dir = os.path.join(base_gallery, person_id)
            os.makedirs(query_dir, exist_ok=True)
            os.makedirs(gallery_dir, exist_ok=True)

            gallery_count = len(glob.glob(os.path.join(gallery_dir, "img*.jpg")))
            print(f"[TEST MODE] Capturing for {person_id}")
            print("First image → query/, rest → gallery/. Press ENTER to save. ESC to stop.")
            cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)

            image_count = 0
            while True:
                frame = self.system.camera.get_frame()
                if frame is None:
                    continue

                person_img, _ = self.system.detector.get_first_person(frame)
                display = person_img if person_img is not None else frame
                cv2.imshow("Capture", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # ENTER
                    if person_img is not None:
                        ext = ".jpg"
                        if image_count == 0:
                            filename = f"img001{ext}"
                            dst = os.path.join(query_dir, filename)
                            print(f"Saved QUERY: {dst}")
                        else:
                            gallery_count += 1
                            filename = f"img{gallery_count:03d}{ext}"
                            dst = os.path.join(gallery_dir, filename)
                            print(f"Saved GALLERY: {dst}")
                        cv2.imwrite(dst, person_img)
                        image_count += 1
                elif key == 27:  # ESC
                    print(f"[DONE] Saved 1 query, {image_count - 1} gallery for {person_id}")
                    cv2.destroyWindow("Capture")
                    break

        else:
            print("[ERROR] Invalid mode selected.")

    def test_and_finetune_menu(self):
        """Menu for testing and fine-tuning operations."""
        while True:
            print("\n=== Model Testing & Fine-tuning ===")
            print("1. Test current model accuracy")
            print("2. Fine-tune model on custom dataset") 
            print("3. Compare before/after fine-tuning")
            print("4. View dataset statistics")
            print("5. Back to main menu")
            
            choice = input("Select an option: ").strip()
            
            if choice == "1":
                self.test_current_model()
            elif choice == "2":
                self.finetune_model()
            elif choice == "3":
                self.compare_models()
            elif choice == "4":
                self.show_dataset_stats()
            elif choice == "5":
                break
            else:
                print("Invalid choice.")

    def test_current_model(self):
        """Test the current model's accuracy."""
        print("\n=== Testing Current Model ===")
        
        # Check if test dataset exists
        if not os.path.exists("../dataset/query") or not os.path.exists("../dataset/gallery"):
            print("Error: No test dataset found!")
            print("Please collect test dataset first using option 3 -> mode 2")
            return
        
        try:
            tester = ModelTester(self.system.recognizer.feature_extractor)
            
            # Get confidence threshold from user
            threshold = input("Enter confidence threshold (default 0.5): ").strip()
            threshold = float(threshold) if threshold else 0.5
            
            print("Testing model... This may take a few minutes.")
            results = tester.test_model_accuracy(confidence_threshold=threshold)
            
            # Save results with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f"../dataset/test_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {results_file}")
            
        except Exception as e:
            print(f"Testing failed: {e}")

    def finetune_model(self):
        """Fine-tune the model on custom dataset."""
        print("\n=== Fine-tuning Model ===")
        
        # Check if training dataset exists
        if not os.path.exists("../dataset/train") or not os.path.exists("../dataset/val"):
            print("Error: No training dataset found!")
            print("Please collect training dataset first using option 3 -> mode 1")
            return
        
        # Show dataset info
        self.show_dataset_stats()
        
        try:
            tester = ModelTester(self.system.recognizer.feature_extractor)
            
            # Get training parameters
            print("\nTraining Parameters:")
            epochs = input("Number of epochs (default 10): ").strip()
            epochs = int(epochs) if epochs else 10
            
            lr = input("Learning rate (default 0.0001): ").strip()
            lr = float(lr) if lr else 0.0001
            
            batch_size = input("Batch size (default 16): ").strip()
            batch_size = int(batch_size) if batch_size else 16
            
            print(f"\nStarting fine-tuning with:")
            print(f"  Epochs: {epochs}")
            print(f"  Learning Rate: {lr}")
            print(f"  Batch Size: {batch_size}")
            print("\nThis may take several minutes...")
            
            # Test before fine-tuning (if test data exists)
            before_results = None
            if os.path.exists("../dataset/query") and os.path.exists("../dataset/gallery"):
                print("\nTesting model before fine-tuning...")
                before_results = tester.test_model_accuracy()
            
            # Fine-tune
            history = tester.fine_tune_model(
                epochs=epochs, 
                learning_rate=lr, 
                batch_size=batch_size
            )
            
            # Test after fine-tuning
            after_results = None
            if os.path.exists("../dataset/query") and os.path.exists("../dataset/gallery"):
                print("\nTesting model after fine-tuning...")
                after_results = tester.test_model_accuracy()
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if before_results and after_results:
                comparison = {
                    "before_finetuning": before_results,
                    "after_finetuning": after_results,
                    "improvement": {
                        "accuracy": after_results["accuracy"] - before_results["accuracy"],
                        "avg_similarity": after_results["average_similarity"] - before_results["average_similarity"]
                    }
                }
                
                comparison_file = f"../dataset/comparison_{timestamp}.json"
                with open(comparison_file, "w") as f:
                    json.dump(comparison, f, indent=2)
                
                print(f"\n=== Fine-tuning Results ===")
                print(f"Accuracy before: {before_results['accuracy']:.4f}")
                print(f"Accuracy after:  {after_results['accuracy']:.4f}")
                print(f"Improvement:     {comparison['improvement']['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")

    def compare_models(self):
        """Compare model performance before and after fine-tuning."""
        print("\n=== Model Comparison ===")
        
        # Find latest comparison file
        comparison_files = glob.glob("../dataset/comparison_*.json")
        if not comparison_files:
            print("No comparison data found. Please run fine-tuning first.")
            return
        
        # Get the most recent comparison
        latest_file = max(comparison_files, key=os.path.getctime)
        
        try:
            with open(latest_file, "r") as f:
                comparison = json.load(f)
            
            print(f"\nOverall Performance:")
            print(f"  Before Fine-tuning - Accuracy: {comparison['before_finetuning']['accuracy']:.4f}")
            print(f"  After Fine-tuning  - Accuracy: {comparison['after_finetuning']['accuracy']:.4f}")
            print(f"  Improvement: +{comparison['improvement']['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error reading comparison data: {e}")

    def show_dataset_stats(self):
        """Show statistics about the current dataset."""
        print("\n=== Dataset Statistics ===")
        
        dataset_dirs = {
            "Training": "../dataset/train",
            "Validation": "../dataset/val", 
            "Query": "../dataset/query",
            "Gallery": "../dataset/gallery"
        }
        
        for split_name, split_path in dataset_dirs.items():
            if os.path.exists(split_path):
                person_dirs = [d for d in os.listdir(split_path) 
                            if os.path.isdir(os.path.join(split_path, d)) and d.startswith('person_')]
                
                total_images = 0
                for person_dir in person_dirs:
                    person_path = os.path.join(split_path, person_dir)
                    image_files = glob.glob(os.path.join(person_path, "*.jpg")) + \
                                glob.glob(os.path.join(person_path, "*.png"))
                    total_images += len(image_files)
                
                print(f"{split_name}: {len(person_dirs)} persons, {total_images} images")
            else:
                print(f"{split_name}: Not found")