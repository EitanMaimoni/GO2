import glob
import json
import os
import time
import cv2
import numpy as np
from models.model_tester import ModelTester

class FinetuneOSNet:

    def __init__(self, system):
        """
        Initialize the OSNet fine-tuning system.
        
        Args:
            system: The main system instance containing camera, detector, recognizer, etc.
        """
        self.system = system
        self.before_finetune_results = None

    def run(self):
        print("\n=== OSNet Fine-Tuning System ===")
        print("1. Collect Fine-Tuning Dataset")
        print("2. Test and Fine-Tune Model")

        choice = input("Select an option: ").strip()

        if choice == "1":
            self.collect_finetune_dataset()
        elif choice == "2":
            self.test_and_finetune_menu()
        else:
            print("[ERROR] Invalid choice. Please select 1 or 2.")

    def collect_finetune_dataset(self):
        print("\n=== Collect Fine-Tuning Dataset ===")
        print("1. Add person for training")
        print("2. Add person for testing")
        mode = input("Choose mode (1/2): ").strip()

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
            print("3. View dataset statistics")
            print("4. Back to main menu")
            
            choice = input("Select an option: ").strip()
            
            if choice == "1":
                self.test_current_model()
            elif choice == "2":
                self.finetune_model()
            elif choice == "3":
                self.show_dataset_stats()
            elif choice == "4":
                break
            else:
                print("Invalid choice.")

    def test_current_model(self):
        """Test the current model's accuracy for each person individually."""
        print("\n=== Testing Current Model ===")
        
        query_dir = "../dataset/query"
        gallery_dir = "../dataset/gallery"
        
        if not os.path.exists(query_dir) or not os.path.exists(gallery_dir):
            print("Error: No test dataset found!")
            print("Please collect test dataset first using option 3 -> mode 2")
            return
        
        try:
            tester = ModelTester(self.system.recognizer.feature_extractor)
            
            # Get all persons in query directory
            query_persons = [d for d in os.listdir(query_dir) 
                           if os.path.isdir(os.path.join(query_dir, d)) and d.startswith('person_')]
            
            # Get all persons in gallery directory
            gallery_persons = [d for d in os.listdir(gallery_dir) 
                             if os.path.isdir(os.path.join(gallery_dir, d)) and d.startswith('person_')]
            
            if not query_persons:
                print("No persons found in query directory!")
                return
            
            if not gallery_persons:
                print("No persons found in gallery directory!")
                return
            
            print(f"Found {len(query_persons)} persons in query and {len(gallery_persons)} persons in gallery")
            
            # Results storage
            detailed_results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "per_person_results": {},
                "overall_stats": {}
            }
            
            all_similarities_same = []  # Similarities when comparing same person
            all_similarities_diff = []  # Similarities when comparing different persons
            
            print("\n" + "="*80)
            print("DETAILED PERSON-BY-PERSON TESTING")
            print("="*80)
            
            for query_person in sorted(query_persons):
                print(f"\n--- Testing {query_person} ---")
                
                query_person_dir = os.path.join(query_dir, query_person)
                query_images = glob.glob(os.path.join(query_person_dir, "*.jpg")) + \
                              glob.glob(os.path.join(query_person_dir, "*.png"))
                
                if not query_images:
                    print(f"  No images found for {query_person}")
                    continue
                
                person_results = {
                    "similarities_to_self": [],
                    "similarities_to_others": {},
                    "best_matches": [],
                    "correct_identifications": 0,
                    "total_queries": len(query_images)
                }
                
                # Test each query image of this person
                for query_img_path in query_images:
                    query_img = cv2.imread(query_img_path)
                    if query_img is None:
                        continue
                    
                    query_feature = self.system.recognizer.feature_extractor.extract(query_img)
                    if query_feature is None:
                        continue
                    
                    best_similarity = -1
                    best_match_person = None
                    person_similarities = {}
                    
                    # Compare against all gallery persons
                    for gallery_person in gallery_persons:
                        gallery_person_dir = os.path.join(gallery_dir, gallery_person)
                        gallery_images = glob.glob(os.path.join(gallery_person_dir, "*.jpg")) + \
                                       glob.glob(os.path.join(gallery_person_dir, "*.png"))
                        
                        person_max_sim = -1
                        
                        for gallery_img_path in gallery_images:
                            gallery_img = cv2.imread(gallery_img_path)
                            if gallery_img is None:
                                continue
                            
                            gallery_feature = self.system.recognizer.feature_extractor.extract(gallery_img)
                            if gallery_feature is None:
                                continue
                            
                            # Calculate similarity
                            from sklearn.metrics.pairwise import cosine_similarity
                            similarity = cosine_similarity(query_feature, gallery_feature)[0][0]
                            person_max_sim = max(person_max_sim, similarity)
                        
                        if person_max_sim > -1:
                            person_similarities[gallery_person] = person_max_sim
                            
                            if person_max_sim > best_similarity:
                                best_similarity = person_max_sim
                                best_match_person = gallery_person
                    
                    # Store results for this query image
                    if best_match_person:
                        person_results["best_matches"].append({
                            "query_image": os.path.basename(query_img_path),
                            "best_match": best_match_person,
                            "similarity": best_similarity,
                            "is_correct": best_match_person == query_person
                        })
                        
                        if best_match_person == query_person:
                            person_results["correct_identifications"] += 1
                            all_similarities_same.append(best_similarity)
                        else:
                            all_similarities_diff.append(best_similarity)
                    
                    # Store similarities to self and others
                    if query_person in person_similarities:
                        person_results["similarities_to_self"].append(person_similarities[query_person])
                    
                    for other_person, sim in person_similarities.items():
                        if other_person != query_person:
                            if other_person not in person_results["similarities_to_others"]:
                                person_results["similarities_to_others"][other_person] = []
                            person_results["similarities_to_others"][other_person].append(sim)
                
                # Calculate and display results for this person
                accuracy = (person_results["correct_identifications"] / 
                           person_results["total_queries"]) if person_results["total_queries"] > 0 else 0
                
                avg_sim_to_self = np.mean(person_results["similarities_to_self"]) \
                                 if person_results["similarities_to_self"] else 0
                
                print(f"  Accuracy: {accuracy:.2%} ({person_results['correct_identifications']}/{person_results['total_queries']})")
                print(f"  Average similarity to self: {avg_sim_to_self:.4f}")
                
                if person_results["similarities_to_others"]:
                    print("  Average similarities to others:")
                    for other_person, sims in person_results["similarities_to_others"].items():
                        avg_sim = np.mean(sims)
                        print(f"    vs {other_person}: {avg_sim:.4f}")
                
                detailed_results["per_person_results"][query_person] = person_results
            
            # Calculate overall statistics
            total_queries = sum(results["total_queries"] for results in detailed_results["per_person_results"].values())
            total_correct = sum(results["correct_identifications"] for results in detailed_results["per_person_results"].values())
            overall_accuracy = total_correct / total_queries if total_queries > 0 else 0
            
            detailed_results["overall_stats"] = {
                "overall_accuracy": overall_accuracy,
                "total_queries": total_queries,
                "total_correct": total_correct,
                "avg_similarity_same_person": np.mean(all_similarities_same) if all_similarities_same else 0,
                "avg_similarity_different_person": np.mean(all_similarities_diff) if all_similarities_diff else 0,
                "similarity_separation": np.mean(all_similarities_same) - np.mean(all_similarities_diff) if all_similarities_same and all_similarities_diff else 0
            }
            
            print("\n" + "="*80)
            print("OVERALL RESULTS")
            print("="*80)
            print(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_queries})")
            if all_similarities_same:
                print(f"Average similarity to same person: {np.mean(all_similarities_same):.4f}")
            if all_similarities_diff:
                print(f"Average similarity to different persons: {np.mean(all_similarities_diff):.4f}")
            if all_similarities_same and all_similarities_diff:
                separation = np.mean(all_similarities_same) - np.mean(all_similarities_diff)
                print(f"Similarity separation (higher is better): {separation:.4f}")
            
            # Save detailed results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = f"../dataset/detailed_test_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(detailed_results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {results_file}")
            
            # Store for comparison
            self.before_finetune_results = detailed_results
            
        except Exception as e:
            print(f"Testing failed: {e}")
            import traceback
            traceback.print_exc()

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
            
            # Test before fine-tuning if not already done
            if self.before_finetune_results is None:
                print("\nTesting model before fine-tuning...")
                self.test_current_model()
            
            # Fine-tune
            history = tester.fine_tune_model(
                epochs=epochs, 
                learning_rate=lr, 
                batch_size=batch_size
            )
            
            print("\nFine-tuning completed successfully!")
            print("You can now test the improved model using option 1, then compare results using option 3.")
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()

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
                print(f"\n{split_name}:")
                for person_dir in sorted(person_dirs):
                    person_path = os.path.join(split_path, person_dir)
                    image_files = glob.glob(os.path.join(person_path, "*.jpg")) + \
                                glob.glob(os.path.join(person_path, "*.png"))
                    person_image_count = len(image_files)
                    total_images += person_image_count
                    print(f"  {person_dir}: {person_image_count} images")
                
                print(f"  Total: {len(person_dirs)} persons, {total_images} images")
            else:
                print(f"{split_name}: Not found")