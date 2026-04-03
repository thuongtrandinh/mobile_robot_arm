"""
Appearance Feature Extractor for Re-ID (Re-identification)
Extracts features from detection crops for BoT-SORT tracker
Maintains appearance features only for actively tracked objects
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class FeatureExtractor:
    """
    Extract appearance features from image crops using CNN
    Uses ResNet backbone optimized for Re-identification
    """
    
    def __init__(self, feature_dim: int = 256, use_gpu: bool = True):
        """
        Args:
            feature_dim: Dimension of output features
            use_gpu: Use GPU if available
        """
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self._build_model()
    
    def _build_model(self):
        """Build lightweight feature extractor"""
        try:
            # Try to use ResNet-18 from torchvision
            import torchvision.models as models
            model = models.resnet18(weights=None)
            # Remove classification head
            self.model = nn.Sequential(*list(model.children())[:-1])
            # Add projection layer to feature_dim
            self.projection = nn.Linear(512, self.feature_dim)
        except:
            # Fallback: simple CNN
            self.model = self._build_simple_cnn()
            self.projection = None
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.projection:
            self.projection = self.projection.to(self.device)
            self.projection.eval()
    
    def _build_simple_cnn(self) -> nn.Module:
        """Lightweight CNN for environments without torchvision"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract features from detections in frame
        
        Args:
            frame: Input image (BGR)
            bboxes: Nx4 array of [x1, y1, x2, y2]
        
        Returns:
            Nx256 array of appearance features
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim))
        
        features = []
        
        with torch.no_grad():
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                # Crop region
                crop = frame[y1:y2, x1:x2]
                
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    features.append(np.zeros(self.feature_dim))
                    continue
                
                # Preprocess
                crop_tensor = self._preprocess(crop)
                
                # Extract feature
                feat = self.model(crop_tensor)
                
                if self.projection:
                    feat = self.projection(feat)
                
                feat_np = feat.cpu().numpy().flatten()
                
                # Normalize
                feat_np = feat_np / (np.linalg.norm(feat_np) + 1e-5)
                
                features.append(feat_np)
        
        return np.array(features)
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model"""
        # Resize to 128x64 (standard Re-ID size)
        image = cv2.resize(image, (64, 128))
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # To tensor
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        return tensor


class FeatureExtractorLightweight:
    """
    Ultra-lightweight feature extractor using color histograms
    Optimized for real-time performance with minimal CPU overhead
    """
    
    def __init__(self, n_bins: int = 32):
        """
        Args:
            n_bins: Histogram bins (32 = balanced speed/accuracy)
        """
        self.n_bins = n_bins
        self.feature_dim = n_bins * 3  # RGB: 96-D feature vector
    
    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from bounding boxes
        
        Args:
            frame: Input image (BGR)
            bboxes: Nx4 array of [x1, y1, x2, y2]
        
        Returns:
            NxF array of histogram features (F = n_bins * 3)
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim))
        
        features = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            crop = frame[y1:y2, x1:x2]
            
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                features.append(np.zeros(self.feature_dim))
                continue
            
            # Extract color histograms for each channel
            feature = np.concatenate([
                cv2.calcHist([crop], [i], None, [self.n_bins], [0, 256]).flatten()
                for i in range(3)  # B, G, R
            ])
            
            # Normalize to probability distribution
            feature = feature / (np.sum(feature) + 1e-5)
            
            features.append(feature)
        
        return np.array(features)
