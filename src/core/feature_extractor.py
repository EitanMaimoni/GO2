import torch
import numpy as np
import cv2
from torchvision import transforms
from torchreid.models.osnet import osnet_x1_0

class FeatureExtractor:
    """Extracts 512-dim features from person images using OSNet."""

    def __init__(self):
        # Chooses whether to use GPU (cuda) or CPU based on availability.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} in use")
        else:
            print("[INFO] Using CPU")

        # Auto-download pretrained weights
        self.model = osnet_x1_0(pretrained=True)
        self.model.to(self.device)
        # Sets the model to evaluation mode (disables dropout, freezes batchnorm)
        self.model.eval()

        # IMPORTANT: Standard transform for OSNet - do not modify these parameters
        # The pretrained model expects images to be resized to 256x128 and normalized 
        # with ImageNet mean and std. Changing these values will negatively impact performance.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),                           
            transforms.Resize((256, 128)),                      
            transforms.ToTensor(),                              
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   
                                 std=[0.229, 0.224, 0.225])    
        ])

    def extract(self, image):
        try:
            if image is None or image.shape[0] < 10 or image.shape[1] < 10:
                return None

            # Convert BGR to RGB (OpenCV uses BGR, PyTorch expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

           # Apply transforms
            img_tensor = self.transform(image)
            # Add batch dimension: [3, 256, 128] → [1, 3, 256, 128] as PyTorch expects
            img_tensor = img_tensor.unsqueeze(0)
            # Move tensor to same device as model (CPU or GPU)
            img_tensor = img_tensor.to(self.device)

            # Disable gradient calculation for inference (saves memory and improves speed)
            with torch.no_grad():
                # Run forward pass through OSNet model
                features = self.model(img_tensor)

            # Move result to CPU and convert to NumPy array: shape = (1, 512)
            return features.cpu().numpy()

        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None
