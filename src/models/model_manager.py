import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
from datetime import datetime
from sklearn.decomposition import PCA

class ModelManager:
    """Manages person recognition models."""
    
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
        Set feature extractor.
        
        Args:
            extractor: FeatureExtractor instance
        """
        self.feature_extractor = extractor
    
    def create_dataset(self, person_name):
        """
        Create directory structure for a new person dataset.
        
        Args:
            person_name: Name/identifier for the person
        """
        person_dir = os.path.join(self.base_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        os.makedirs(os.path.join(person_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(person_dir, "model"), exist_ok=True)
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(person_dir, "raw", f"{timestamp}.jpg")
        cv2.imwrite(filename, image)
        return filename
    
    def list_models(self):
        """
        List all available trained person models.
        
        Returns:
            list: Names of all available person models
        """
        models = []
        for name in os.listdir(self.base_dir):
            person_dir = os.path.join(self.base_dir, name)
            model_dir = os.path.join(person_dir, "model")
            if os.path.isdir(person_dir) and os.path.isdir(model_dir):
                model_path = os.path.join(model_dir, f"{name}_model.pkl")
                pca_path = os.path.join(model_dir, f"{name}_pca.pkl")
                if os.path.exists(model_path) and os.path.exists(pca_path):
                    models.append(name)
        return models
    
    def load_model(self, person_name):
        """
        Load a trained model for a specific person.
        
        Args:
            person_name: Name of the person whose model to load
            
        Returns:
            tuple: (clf, pca) where:
                - clf: Trained OneClassSVM model
                - pca: PCA transformation used during training
                
        Raises:
            FileNotFoundError: If no model exists for specified person
        """
        model_path = os.path.join(self.base_dir, person_name, "model", f"{person_name}_model.pkl")
        pca_path = os.path.join(self.base_dir, person_name, "model", f"{person_name}_pca.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(f"No complete model found for {person_name}")
            
        clf = joblib.load(model_path)
        pca = joblib.load(pca_path)
        
        return clf, pca
    
    def train_model(self, person_name):
        """
        Train a one-class recognition model for a specific person.
        
        Args:
            person_name: Name of the person to train model for
            
        Returns:
            str: Path to the saved model file
            
        Raises:
            ValueError: If not enough valid images or other training error
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set")
            
        person_dir = os.path.join(self.base_dir, person_name)
        raw_dir = os.path.join(person_dir, "raw")
        model_dir = os.path.join(person_dir, "model")
        
        # Get all positive images
        image_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.jpg')]
        
        if len(image_files) < self.min_images:
            raise ValueError(f"Need at least {self.min_images} images for training, only found {len(image_files)}")
        
        # Extract features
        features = []
        for image_file in image_files:
            image = cv2.imread(image_file)
            if image is None:
                continue
            feature = self.feature_extractor.extract_features(image)
            if feature is not None:
                features.append(feature)
        
        if len(features) < self.min_images:
            raise ValueError(f"Not enough valid images for training, need {self.min_images}, got {len(features)}")
        
        # Convert to numpy array
        X = np.array(features)
        
        # Apply PCA
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        X_reduced = pca.fit_transform(X)
        
        # Train One-Class SVM
        clf = OneClassSVM(kernel='rbf', nu=0.1)  # nu is the outlier fraction
        clf.fit(X_reduced)
        
        # Save model
        model_path = os.path.join(model_dir, f"{person_name}_model.pkl")
        pca_path = os.path.join(model_dir, f"{person_name}_pca.pkl")
        
        joblib.dump(clf, model_path)
        joblib.dump(pca, pca_path)
        
        return model_path