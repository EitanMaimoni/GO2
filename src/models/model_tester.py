import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchreid.models.osnet import osnet_x1_0
from PIL import Image
import glob


class PersonDataset(Dataset):
    """Dataset for fine-tuning person re-identification models."""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Args:
            data_dir: Path to dataset directory (should contain train/val or query/gallery subdirs)
            transform: Transformations to apply to images
            mode: 'train', 'val', 'query', or 'gallery'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.samples = []
        self.class_to_idx = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load data paths and create class mappings."""
        mode_dir = os.path.join(self.data_dir, self.mode)
        
        if not os.path.exists(mode_dir):
            raise ValueError(f"Directory {mode_dir} does not exist")
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(mode_dir) 
                      if os.path.isdir(os.path.join(mode_dir, d)) and d.startswith('person_')]
        
        # Create class to index mapping
        self.class_to_idx = {person_id: idx for idx, person_id in enumerate(sorted(person_dirs))}
        
        # Load all image paths with their labels
        for person_id in person_dirs:
            person_path = os.path.join(mode_dir, person_id)
            image_files = glob.glob(os.path.join(person_path, "*.jpg")) + \
                         glob.glob(os.path.join(person_path, "*.png"))
            
            label = self.class_to_idx[person_id]
            for img_path in image_files:
                self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ModelTester:
    """Class for testing and fine-tuning person re-identification models."""
    
    def __init__(self, feature_extractor, dataset_path="../dataset"):
        """
        Args:
            feature_extractor: Your existing FeatureExtractor instance
            dataset_path: Path to the dataset directory
        """
        self.feature_extractor = feature_extractor
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Standard OSNet transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def test_model_accuracy(self, confidence_threshold=0.5):
        """
        Test the current model's accuracy using query/gallery evaluation.
        
        Args:
            confidence_threshold: Threshold for positive identification
            
        Returns:
            dict: Testing results including accuracy, precision, recall
        """
        print("\n=== Testing Model Accuracy ===")
        
        query_dir = os.path.join(self.dataset_path, "query")
        gallery_dir = os.path.join(self.dataset_path, "gallery")
        
        if not os.path.exists(query_dir) or not os.path.exists(gallery_dir):
            raise ValueError("Query and gallery directories must exist for testing")
        
        # Load query and gallery data
        query_dataset = PersonDataset(self.dataset_path, self.transform, mode='query')
        gallery_dataset = PersonDataset(self.dataset_path, self.transform, mode='gallery')
        
        query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
        gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)
        
        # Extract features for all gallery images
        print("Extracting gallery features...")
        gallery_features = []
        gallery_labels = []
        
        for images, labels in gallery_loader:
            for i in range(len(images)):
                img_np = self._tensor_to_numpy(images[i])
                feature = self.feature_extractor.extract(img_np)
                if feature is not None:
                    gallery_features.append(feature[0])  # Remove batch dimension
                    gallery_labels.append(labels[i].item())
        
        gallery_features = np.array(gallery_features)
        gallery_labels = np.array(gallery_labels)
        
        # Test query images against gallery
        print("Testing query images...")
        predictions = []
        true_labels = []
        similarities_list = []
        
        for images, labels in query_loader:
            for i in range(len(images)):
                img_np = self._tensor_to_numpy(images[i])
                query_feature = self.feature_extractor.extract(img_np)
                
                if query_feature is not None:
                    # Compute similarities with all gallery images
                    similarities = cosine_similarity(query_feature, gallery_features)[0]
                    max_similarity = np.max(similarities)
                    best_match_idx = np.argmax(similarities)
                    
                    # Predict based on threshold
                    if max_similarity >= confidence_threshold:
                        predicted_label = gallery_labels[best_match_idx]
                    else:
                        predicted_label = -1  # Unknown person
                    
                    predictions.append(predicted_label)
                    true_labels.append(labels[i].item())
                    similarities_list.append(max_similarity)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate precision and recall for each class
        unique_labels = sorted(set(true_labels))
        results = {
            'accuracy': accuracy,
            'total_queries': len(true_labels),
            'correct_predictions': sum(1 for t, p in zip(true_labels, predictions) if t == p),
            'average_similarity': np.mean(similarities_list),
            'per_class_results': {}
        }
        
        for label in unique_labels:
            true_pos = sum(1 for t, p in zip(true_labels, predictions) if t == label and p == label)
            false_pos = sum(1 for t, p in zip(true_labels, predictions) if t != label and p == label)
            false_neg = sum(1 for t, p in zip(true_labels, predictions) if t == label and p != label)
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['per_class_results'][f'person_{label:03d}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': true_pos,
                'false_positives': false_pos,
                'false_negatives': false_neg
            }
        
        self._print_test_results(results)
        return results
    
    def fine_tune_model(self, epochs=10, learning_rate=0.0001, batch_size=16):
        """
        Fine-tune the OSNet model on your custom dataset.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            batch_size: Training batch size
            
        Returns:
            dict: Training history
        """
        print("\n=== Fine-tuning Model ===")
        
        # Create datasets
        train_dataset = PersonDataset(self.dataset_path, self.transform, mode='train')
        val_dataset = PersonDataset(self.dataset_path, self.transform, mode='val')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create a new model for fine-tuning
        model = osnet_x1_0(num_classes=len(train_dataset.class_to_idx))
        
        # Load pretrained weights (except classifier layer)
        pretrained_model = osnet_x1_0(pretrained=True)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        # Filter out classifier layer
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            scheduler.step()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_acc)
            
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save fine-tuned model
        model_save_path = os.path.join(self.dataset_path, "fine_tuned_osnet.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Fine-tuned model saved to: {model_save_path}")
        
        # Update the feature extractor with fine-tuned model
        self.feature_extractor.model.load_state_dict(model.state_dict())
        print("Feature extractor updated with fine-tuned weights")
        
        return history
    
    def _tensor_to_numpy(self, tensor):
        """Convert tensor back to numpy array in OpenCV format."""
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img = tensor.numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def _print_test_results(self, results):
        """Print formatted test results."""
        print(f"\n=== Test Results ===")
        print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['correct_predictions']}/{results['total_queries']})")
        print(f"Average Similarity: {results['average_similarity']:.4f}")
        print("\nPer-Class Results:")
        
        for person_id, metrics in results['per_class_results'].items():
            print(f"{person_id}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    def plot_training_history(self, history):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_path, 'training_history.png'))
        plt.show()