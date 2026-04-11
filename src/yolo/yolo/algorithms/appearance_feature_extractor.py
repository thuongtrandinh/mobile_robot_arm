"""
Deep Learning Appearance Extractor for Re-ID (Re-identification)
Optimized for NVIDIA RTX A4000 GPU with Tensor Cores (FP16 acceleration)

Uses ResNet18 pre-trained backbone to extract 512-dimensional feature vectors
that capture person identity, clothing texture, posture, and silhouette.

[ĐÃ TỐI ƯU HÓA ẢNH XÁM & HÌNH HỌC]: Tích hợp bộ lọc Local CLAHE và 
Geometry Filter ngay trong class để loại bỏ các crop lỗi/rác trước khi đưa vào GPU.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class FeatureExtractorDeep:
    """
    Deep Learning Appearance Extractor using ResNet18 (Tensor Cores optimized)
    """
    
    def __init__(self, device='cuda', half=True):
        self.device = device
        self.half = half
        self.feature_dim = 512  # ResNet18 outputs 512-dim features
        
        # Load pre-trained ResNet18 (ImageNet weights)
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        
        # Remove classification head
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        
        # Enable Tensor Core FP16 for RTX A4000
        if self.half:
            self.model.half()
        
        self.model.eval()
        
        # ImageNet normalization
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((256, 128), antialias=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Bộ lọc CLAHE khôi phục chi tiết bề mặt
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def get_empty_feature(self):
        return np.zeros((self.feature_dim,), dtype=np.float32)
    
    def extract(self, img_crop):
        """
        Extract 512-dim ReID feature from a person crop image.
        """
        if img_crop is None or img_crop.size == 0:
            return self.get_empty_feature()
        
        # Lấy kích thước thực tế của Bounding Box từ chính crop_img
        h, w = img_crop.shape[:2]
        
        # =================================================================
        # 1. BỘ LỌC HÌNH HỌC (GEOMETRY FILTER) - TỐI ƯU HIỆU NĂNG GPU
        # =================================================================
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Bỏ qua các Box quá thấp (< 100px) hoặc sai tỷ lệ dáng người (Rộng/Cao > 0.85)
        # Giúp tiết kiệm 1.5ms thời gian tính toán cho mỗi vật thể rác
        if h < 100 or aspect_ratio > 0.85 or aspect_ratio < 0.15:
            return self.get_empty_feature()
        
        try:
            # =================================================================
            # 2. BỘ LỌC ẢNH XÁM (GRAYSCALE OPTIMIZATION)
            # =================================================================
            if len(img_crop.shape) == 3 and img_crop.shape[2] == 3:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            elif len(img_crop.shape) == 3 and img_crop.shape[2] == 4:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGRA2GRAY)
            elif len(img_crop.shape) == 2 or img_crop.shape[2] == 1:
                gray = img_crop
            else:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

            enhanced_gray = self.clahe.apply(gray)
            enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

            # =================================================================
            # 3. TRÍCH XUẤT ĐẶC TRƯNG TENSOR CORES
            # =================================================================
            tensor = self.transforms(enhanced_rgb).unsqueeze(0).to(self.device)
            
            if self.half:
                tensor = tensor.half()
            
            with torch.no_grad():
                feat = self.model(tensor)
            
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.cpu().numpy().flatten().astype(np.float32)
        
        except Exception as e:
            print(f"⚠️ Deep Extractor Error: {e}")
            return self.get_empty_feature()


class FeatureExtractorLightweight:
    """
    Lightweight fallback extractor (for low-end GPUs or CPU-only mode)
    """
    
    def __init__(self, n_bins: int = 32):
        self.n_bins = n_bins
        self.feature_dim = n_bins
    
    def get_empty_feature(self):
        return np.zeros((self.feature_dim,), dtype=np.float32)
    
    def extract(self, img_crop):
        if img_crop is None or img_crop.size == 0:
            return self.get_empty_feature()
        
        # Áp dụng bộ lọc hình học tương tự cho CPU mode
        h, w = img_crop.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        
        if h < 100 or aspect_ratio > 0.85 or aspect_ratio < 0.15:
            return self.get_empty_feature()
        
        try:
            if len(img_crop.shape) == 3:
                img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_crop
            
            hist = cv2.calcHist([img_gray], [0], None, [self.n_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            return hist.astype(np.float32)
        
        except Exception as e:
            print(f"Lightweight Extractor Error: {e}")
            return self.get_empty_feature()

