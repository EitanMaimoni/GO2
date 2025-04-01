import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
from sklearn.decomposition import PCA

class AppearanceModelManager:
    def __init__(self, base_dir="../persons"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def list_existing_models(self):
        """List all available person models"""
        models = []
        for name in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, name)):
                models.append(name)
        return models
    
    def create_new_person_dataset(self, person_name):
        """Create directory structure for a new person"""
        person_dir = os.path.join(self.base_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        os.makedirs(os.path.join(person_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(person_dir, "model"), exist_ok=True)
        return person_dir
    
    def save_person_image(self, person_name, image):
        """Save an image to the person's dataset"""
        person_dir = os.path.join(self.base_dir, person_name)
        if not os.path.exists(person_dir):
            self.create_new_person_dataset(person_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(person_dir, "raw", f"{timestamp}.jpg")
        cv2.imwrite(filename, image)
        return filename
    
    def extract_appearance_features(self, image):
        """Extract color and texture features from the full body image"""
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
        """Calculate Histogram of Oriented Gradients"""
        from skimage.feature import hog
        features = hog(gray_image, orientations=bins, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
        return features
    
    def train_person_model(self, person_name):
        """Train a one-class recognition model using only positive samples"""
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
        pca = PCA(n_components=0.95)
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
    
    def load_person_model(self, person_name):
        """Load a trained model for a specific person"""
        model_path = os.path.join(self.base_dir, person_name, "model", f"{person_name}_model.pkl")
        pca_path = os.path.join(self.base_dir, person_name, "model", f"{person_name}_pca.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(f"No complete model found for {person_name}")
            
        clf = joblib.load(model_path)
        pca = joblib.load(pca_path)
        
        return clf, pca