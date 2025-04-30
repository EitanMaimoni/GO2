import os
import cv2
import numpy as np
import joblib
from datetime import datetime
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

class ModelManager:
    """Manager for person recognition models."""
    
    def __init__(self, settings):
        """
        Initialize model manager.
        
        Args:
            settings: Application settings
        """
        self.base_dir = settings.persons_dir
        self.min_images = settings.person_image_min_count
        os.makedirs(self.base_dir, exist_ok=True)
        self.feature_extractor = None  # Will be set externally
    
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
    
    # TODO: Add a delete method to delete a person's model
    def delete_model(self, person_name):
        pass
    
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

