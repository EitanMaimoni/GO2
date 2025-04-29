import torch
import numpy as np
import cv2
from torchvision import transforms
from torchreid.models.osnet import osnet_x1_0

class FeatureExtractor:
    """Extracts 512-dim features from person images using OSNet."""

    def __init__(self, settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Auto-download pretrained weights
        self.model = osnet_x1_0(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

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

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(img_tensor)

            return features.cpu().numpy()

        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None
