import os
import cv2
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
from datetime import datetime
from sklearn.decomposition import PCA

class AppearanceModelManager:
    """
    A class for managing appearance models of persons using One-Class SVM.
    
    Responsibilities:
    1. Create and manage dataset directories for different persons
    2. Save and organize captured images of persons
    3. Train and manage one-class recognition models for person re-identification
    4. Extract appearance features (color + texture) for person recognition
    
    Note: Uses PCA for dimensionality reduction and One-Class SVM for modeling.
    """
    
    def __init__(self, base_dir="../persons"):
        """
        Initialize the appearance model manager.
        
        Args:
            base_dir: Base directory where all person models will be stored
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_new_person_dataset(self, person_name):
        """
        Create directory structure for a new person dataset.
        
        Args:
            person_name: Name/identifier for the person
        """
        person_dir = os.path.join(self.base_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        os.makedirs(os.path.join(person_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(person_dir, "model"), exist_ok=True)

    def save_person_image(self, person_name, image):
        """
        Save a captured person image to the person's dataset.
        
        Args:
            person_name: Name of the person to save image for
            image: Numpy array containing the image to save
        """
        person_dir = os.path.join(self.base_dir, person_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(person_dir, "raw", f"{timestamp}.jpg")
        cv2.imwrite(filename, image)
    
    def list_existing_models(self):
        """
        List all available trained person models.
        
        Returns:
            list: Names of all available person models
        """
        models = []
        for name in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, name)):
                models.append(name)
        return models
    
    def load_person_model(self, person_name):
        """
        Load a trained model for a specific person.
        
        Args:
            person_name: Name of the person whose model to load
            
        Returns:
            tuple: (clf, pca) where:
                - clf: Trained OneClassSVM model
                - pca: PCA transformation used during training
        """
        model_path = os.path.join(self.base_dir, person_name, "model", f"{person_name}_model.pkl")
        pca_path = os.path.join(self.base_dir, person_name, "model", f"{person_name}_pca.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(f"No complete model found for {person_name}")
            
        clf = joblib.load(model_path)
        pca = joblib.load(pca_path)
        
        return clf, pca
    
    def train_person_model(self, person_name):
        """
        Train a one-class recognition model for a specific person.
        
        Args:
            person_name: Name of the person to train model for
            
        Returns:
            str: Path to the saved model file
        """
        person_dir = os.path.join(self.base_dir, person_name)
        raw_dir = os.path.join(person_dir, "raw")
        model_dir = os.path.join(person_dir, "model")
        
        # Get all positive images
        image_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.jpg')]
        
        if len(image_files) < 10:
            raise ValueError(f"Need at least 10 images for training, only found {len(image_files)}")
        
        # Extract features
        features = []
        for image_file in image_files:
            image = cv2.imread(image_file)
            if image is None:
                continue
            feature = self.extract_appearance_features(image)
            features.append(feature)
        
        if not features:
            raise ValueError("No valid images found for training")
        
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

    def extract_appearance_features(self, image):
        """
        Extract color and texture features from a person image.
        
        Args:
            image: Numpy array containing the person image
            
        Returns:
            numpy.ndarray: Combined feature vector containing:
                - HSV color statistics (mean and std)
                - HOG texture features
        """
        # Resize to consistent dimensions
        resized = cv2.resize(image, (128, 256))
        
        # Color features (mean and std of each channel in HSV space)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_mean, h_std = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
        s_mean, s_std = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
        v_mean, v_std = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
        
        # Texture features (using histogram of oriented gradients)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog = self._calculate_hog(gray)
        
        # Combine all features
        features = np.array([
            h_mean, h_std, s_mean, s_std, v_mean, v_std,
            *hog
        ])
        
        return features
    
    def _calculate_hog(self, gray_image, bins=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
        """
        Calculate Histogram of Oriented Gradients (HOG) features.
        
        Args:
            gray_image: Grayscale image
            bins: Number of orientation bins
            pixels_per_cell: Size (in pixels) of a cell
            cells_per_block: Number of cells in each block
            
        Returns:
            numpy.ndarray: HOG feature vector
        """
        from skimage.feature import hog
        features = hog(gray_image, orientations=bins, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
        return features