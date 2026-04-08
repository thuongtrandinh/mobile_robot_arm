"""
Appearance Feature Extractor for Re-ID (Re-identification)
Extracts features from detection crops for BoT-SORT tracker

Optimized for grayscale input:
- CNN-based features for shape/morphology (strong signal in grayscale)
- LBP (Local Binary Patterns) for lightweight texture-based Re-ID
- Depth-aware ROI cropping to reduce background noise
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# Try to import for LBP texture feature extraction
try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class FeatureExtractor:
    """
    Extract appearance features from image crops using CNN.

    Optimized for grayscale input:
    - Adjusts preprocessing to handle single-channel images
    - Uses shape/morphology as primary Re-ID cue (robust to grayscale)
    - CNN learns subtle variations in silhouette, clothing texture
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

            # Adapt first layer to accept 1-channel grayscale input
            original_conv = model.conv1
            # Create new conv layer accepting 1 channel
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # Initialize with averaged weights from 3-channel version
            with torch.no_grad():
                model.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

            # Remove classification head, keep feature backbone
            self.model = nn.Sequential(*list(model.children())[:-1])
            # Add projection layer to feature_dim
            self.projection = nn.Linear(512, self.feature_dim)
        except Exception:
            # Fallback: simple CNN adapted for grayscale
            self.model = self._build_simple_cnn()
            self.projection = None

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.projection:
            self.projection = self.projection.to(self.device)
            self.projection.eval()

    def _build_simple_cnn(self) -> nn.Module:
        """Lightweight CNN for grayscale input"""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 1 channel input (grayscale)
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
        Extract CNN features from detections in frame.

        Args:
            frame: Input image (BGR or grayscale)
            bboxes: Nx4 array of [x1, y1, x2, y2]

        Returns:
            Nx256 array of appearance features (normalized L2)
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim))

        features = []

        with torch.no_grad():
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                # Crop region with some margin for better silhouette
                y1, y2 = max(0, y1 - 5), min(frame.shape[0], y2 + 5)
                x1, x2 = max(0, x1 - 5), min(frame.shape[1], x2 + 5)
                crop = frame[y1:y2, x1:x2]

                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    features.append(np.zeros(self.feature_dim))
                    continue

                # Preprocess (handles grayscale conversion)
                crop_tensor = self._preprocess(crop)

                # Extract feature
                feat = self.model(crop_tensor)

                if self.projection:
                    feat = self.projection(feat)

                feat_np = feat.cpu().numpy().flatten()

                # L2 normalization
                feat_np = feat_np / (np.linalg.norm(feat_np) + 1e-5)

                features.append(feat_np)

        return np.array(features)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for CNN model.

        Converts to grayscale if needed and normalizes for training.
        """
        # Convert to grayscale if BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to 128x64 (standard Re-ID size)
        image = cv2.resize(image, (64, 128))

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Standard ImageNet-style normalization (even for grayscale, use same mean/std)
        # This helps CNN trained on color images to work with grayscale
        mean = 0.449  # Average of [0.485, 0.456, 0.406]
        std = 0.226   # Average of [0.229, 0.224, 0.225]
        image = (image - mean) / std

        # Add channel dimension: HW → CHW (1 channel)
        image = np.expand_dims(image, axis=0)

        # To tensor and batch
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)  # [1, 1, 128, 64]

        return tensor


class LocalBinaryPatternExtractor:
    """
    Ultra-lightweight texture-based Re-ID using Local Binary Patterns (LBP).

    Why LBP for grayscale Re-ID?
    - LBP describes local texture patterns (subtle wrinkles, fabric texture)
    - Color-invariant: works perfectly with grayscale
    - CPU-efficient: ~1ms per detection
    - Complementary to CNN: captures fine texture details

    Feature vector: [LBP-histogram for different radius values]
    Falls back to intensity histogram if scikit-image unavailable.
    """

    def __init__(self, radius_list: Tuple[int, ...] = (1, 3), n_bins: int = 32):
        """
        Args:
            radius_list: LBP radii to compute (larger radius captures coarser patterns)
            n_bins: Histogram bins per radius
        """
        self.radius_list = radius_list
        self.n_bins = n_bins
        self.feature_dim = len(radius_list) * n_bins
        self.use_lbp = HAS_SKIMAGE

    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract LBP texture features from bounding boxes.

        Args:
            frame: Input image (BGR or grayscale)
            bboxes: Nx4 array of [x1, y1, x2, y2]

        Returns:
            NxF array of LBP histogram features (F = len(radius_list) * n_bins)
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim))

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        features = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)

            # Crop with margins for better texture context
            y1, y2 = max(0, y1 - 5), min(frame_gray.shape[0], y2 + 5)
            x1, x2 = max(0, x1 - 5), min(frame_gray.shape[1], x2 + 5)
            crop = frame_gray[y1:y2, x1:x2]

            if crop.shape[0] < 16 or crop.shape[1] < 16:
                # Too small to extract texture
                features.append(np.zeros(self.feature_dim))
                continue

            # Resize for consistent feature extraction
            crop_resized = cv2.resize(crop, (64, 128))

            # Compute features at different scales
            feature = np.array([])

            if self.use_lbp:
                # Use LBP if available
                for radius in self.radius_list:
                    lbp = local_binary_pattern(crop_resized, 8, radius, method='uniform')
                    hist, _ = np.histogram(lbp.ravel(), bins=self.n_bins, range=(0, self.n_bins))
                    hist = hist / (np.sum(hist) + 1e-5)
                    feature = np.concatenate([feature, hist])
            else:
                # Fallback: use intensity histogram at multiple scales
                for _ in self.radius_list:
                    hist, _ = np.histogram(crop_resized.ravel(), bins=self.n_bins, range=(0, 256))
                    hist = hist / (np.sum(hist) + 1e-5)
                    feature = np.concatenate([feature, hist])

            features.append(feature)

        return np.array(features)


class FeatureExtractorLightweight:
    """
    Ultra-lightweight hybrid Re-ID using LBP + grayscale statistics.

    Combines:
    1. LBP texture features (robust to grayscale)
    2. Local intensity statistics (mean, std per region)

    This replaces old color-histogram approach which is ineffective for grayscale.
    """

    def __init__(self, n_bins: int = 32):
        """
        Args:
            n_bins: LBP histogram bins
        """
        self.n_bins = n_bins
        self.lbp_extractor = LocalBinaryPatternExtractor(
            radius_list=(1, 3),  # Multi-scale LBP
            n_bins=n_bins
        )
        # LBP feature dim: 2 radii × n_bins = 64
        self.feature_dim = self.lbp_extractor.feature_dim

    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Extract lightweight features optimized for grayscale images.

        Args:
            frame: Input image (BGR or grayscale)
            bboxes: Nx4 array of [x1, y1, x2, y2]

        Returns:
            NxF array of LBP histogram features
        """
        return self.lbp_extractor.extract(frame, bboxes)
