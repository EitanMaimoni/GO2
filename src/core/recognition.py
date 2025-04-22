import numpy as np
import cv2
from skimage.feature import hog

class FeatureExtractor:
    """Extracts appearance features from person images."""
    
    def __init__(self, settings):
        """
        Initialize feature extractor.
        
        Args:
            settings: Application settings
        """
        self.feature_width = settings.feature_image_width
        self.feature_height = settings.feature_image_height
    
    def extract_features(self, image):
        """
        Extract color and texture features from a person image.
        
        Args:
            image: Numpy array containing the person image
            
        Returns:
            numpy.ndarray: Combined feature vector containing:
                - HSV color statistics (mean and std)
                - HOG texture features
        """
        if image is None or image.size == 0:
            return None
            
        # Resize to consistent dimensions
        resized = cv2.resize(image, (self.feature_width, self.feature_height))
        
        # Color features (mean and std of each channel in HSV space)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_mean, h_std = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
        s_mean, s_std = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
        v_mean, v_std = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
        
        # Texture features (using histogram of oriented gradients)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog_features = self._calculate_hog(gray)
        
        # Combine all features
        features = np.concatenate([
            np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std]),
            hog_features
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
        features = hog(gray_image, orientations=bins, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
        return features

class PersonRecognizer:
    """Recognizes specific persons using trained models."""
    
    def __init__(self, feature_extractor, model_manager):
        """
        Initialize person recognizer.
        
        Args:
            feature_extractor: FeatureExtractor instance
            model_manager: ModelManager instance
        """
        self.feature_extractor = feature_extractor
        self.model_manager = model_manager
        self.target_model = None
        self.target_pca = None
    
    def load_target(self, person_name):
        """
        Load model for target person.
        
        Args:
            person_name: Name of person to recognize
            
        Returns:
            bool: Success or failure
        """
        try:
            self.target_model, self.target_pca = self.model_manager.load_model(person_name)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def is_target_person(self, person_img):
        """
        Check if image matches target person.
        
        Args:
            person_img: Person image to check
            
        Returns:
            bool: True if matches target, False otherwise
        """
        if self.target_model is None or self.target_pca is None:
            return False
            
        features = self.feature_extractor.extract_features(person_img)
        if features is None:
            return False
            
        reduced_features = self.target_pca.transform([features])
        return self.target_model.predict(reduced_features) == 1