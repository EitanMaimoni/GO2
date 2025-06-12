import os
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm
import shutil

class ModelManager:
    """Manager for person recognition models."""
    
    def __init__(self, settings, camera, detector, feature_extractor):
        """
        Initialize model manager.
        
        Args:
            settings: Application settings
        """
        self.camera = camera
        self.detector = detector
        self.base_dir = settings.persons_dir
        self.min_images = settings.person_image_min_count
        os.makedirs(self.base_dir, exist_ok=True)
        self.feature_extractor = feature_extractor

    
    def set_feature_extractor(self, extractor):
        """
        Set the feature extractor for the model manager.
        
        Args:
            extractor: FeatureExtractor instance
        """
        self.feature_extractor = extractor
    
    def create_dataset(self, person_name):
        """
        Create directory structure for a new person dataset.
        
        Args:
            person_name: Name/identifier for the person
            
        Returns:
            str: Path to the created person directory
        """
        person_dir = os.path.join(self.base_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        os.makedirs(os.path.join(person_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(person_dir, "gallery"), exist_ok=True)
        return person_dir
    
    def create_model(self):
        name = input("Enter person name: ").strip()

        self.create_dataset(name)
        self.capture_mode = True
        self.capture_count = 0

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            person_img, _ = self.detector.get_first_person(frame)

            display_img = person_img if person_img is not None else frame
            cv2.imshow("Tracking", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                if person_img is not None:
                    self.save_image(name, person_img)
                    self.capture_count += 1
                    print(f"Saved image {self.capture_count}")
            elif key == 27:  # ESC
                print("Training model...")
                try:
                    self.train_model_osnet(name)
                    print(f"Model trained successfully for {name}.")
                except Exception as e:
                    print(f"Training failed: {e}")
                cv2.destroyWindow("Tracking")
                break

    def delete_model(self):
        """
        Delete a person model and all associated data.
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        models = self.list_models()
        if not models:
            print("No trained models available.")
            return

        print("Available models:")
        for i, name in enumerate(models):
            print(f"{i + 1}. {name}")
        
        idx = int(input("Select model to delete: ")) - 1

        if idx < 0 or idx >= len(models):
            print("Invalid selection.")
            return False
        
        name = models[idx]
        person_dir = os.path.join(self.base_dir, name)
        
        try:
            shutil.rmtree(person_dir)
            print(f"Model {name} deleted successfully.")
            return True
        except Exception as e:
            print(f"Failed to delete model {name}: {e}")
            return False

    def save_image(self, person_name, image):
        """
        Save a captured person image to the person's dataset.
        
        Args:
            person_name: Name of the person to save image for
            image: Numpy array containing the image to save
            
        Returns:
            str: Path to saved image
        """
        person_dir = os.path.join(self.base_dir, person_name)
        raw_dir = os.path.join(person_dir, "raw")
        
        # Create directories if they don't exist
        if not os.path.exists(raw_dir):
            self.create_dataset(person_name)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(raw_dir, f"{timestamp}.jpg")
        cv2.imwrite(filename, image)
        return filename
    
    def list_models(self):
        """
        List all persons who have saved feature galleries.
        """
        models = []
        for name in os.listdir(self.base_dir):
            gallery_path = os.path.join(self.base_dir, name, "gallery", "features.npy")
            if os.path.exists(gallery_path):
                models.append(name)
        return models
    
    
    
    # TODO: Maybe its better to blur the background (or something similar)
    def train_model_osnet(self, person_name):
        """
        Extract OSNet features from images of the person and save to gallery.
        """
        assert self.feature_extractor is not None, "Feature extractor not set."

        person_dir = os.path.join(self.base_dir, person_name)
        raw_dir = os.path.join(person_dir, "raw")
        gallery_dir = os.path.join(person_dir, "gallery")
        os.makedirs(gallery_dir, exist_ok=True)

        feature_list = []

        print(f"[INFO] Extracting OSNet features for {person_name}...")

        for filename in tqdm(os.listdir(raw_dir)):
            path = os.path.join(raw_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            feature = self.feature_extractor.extract(img)
            if feature is not None:
                feature_list.append(feature)

        if len(feature_list) < self.min_images:
            raise Exception(f"Not enough valid images for {person_name}.")

        features = np.vstack(feature_list)  # shape: (N, 512)
        
        # Save raw features to gallery
        gallery_path = os.path.join(gallery_dir, "features.npy")
        np.save(gallery_path, features)

        print(f"[SUCCESS] Gallery saved for {person_name}.")

