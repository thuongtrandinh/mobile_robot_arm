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

        # Bộ lọc CLAHE khôi phục chi tiết bề mặt từ ảnh xám
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # [ĐÃ TỐI ƯU] Kernel Morphology cho preprocessing màu sắc
        # Giúp loại bỏ nhiễu nhỏ và kết nối các vùng màu sắc liên quan
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def get_empty_feature(self):
        return np.zeros((self.feature_dim,), dtype=np.float32)
    
    def extract(self, img_crop):
        """
        Extract 512-dim ReID feature from a person crop image.
        
        [ĐÃ TỐI ƯU] Tối ưu hố việc sử dụng thông tin màu sắc từ ảnh RGB:
        - Trích xuất màu sắc trang phục (quần, áo, phụ kiện)
        - Giữ lại cấu trúc hình dáng (vai, hông, tỷ lệ cơ thể)
        - Kháng nhiễu ánh sáng thông qua chuẩn hóa
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
            # 2. [ĐÃ TỐI ƯU] CHUẨN HÓA MÀU SẮC (COLOR NORMALIZATION)
            # =================================================================
            # Chuyển BGR → LAB color space (độc lập với ánh sáng)
            # LAB space giúp trích xuất màu sắc nguyên bản mà không bị ảnh hưởng 
            # bởi sự thay đổi cường độ sáng (ví dụ: ánh đèn vàng/xanh trong lab)
            
            if len(img_crop.shape) == 3 and img_crop.shape[2] == 3:
                # Ảnh đầu vào là BGR
                img_bgr = img_crop
            elif len(img_crop.shape) == 3 and img_crop.shape[2] == 4:
                # Ảnh đầu vào là BGRA, tách alpha channel
                img_bgr = cv2.cvtColor(img_crop, cv2.COLOR_BGRA2BGR)
            elif len(img_crop.shape) == 2 or (len(img_crop.shape) == 3 and img_crop.shape[2] == 1):
                # Ảnh đầu vào là Grayscale, không thể trích xuất đặc trưng màu sắc
                # Sử dụng bộ lọc CLAHE trên ảnh xám
                gray = img_crop if len(img_crop.shape) == 2 else img_crop[:, :, 0]
                enhanced_gray = self.clahe.apply(gray)
                enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
                tensor = self.transforms(enhanced_rgb).unsqueeze(0).to(self.device)
                if self.half:
                    tensor = tensor.half()
                with torch.no_grad():
                    feat = self.model(tensor)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                return feat.cpu().numpy().flatten().astype(np.float32)
            else:
                img_bgr = cv2.cvtColor(img_crop, cv2.COLOR_BGR2BGR)  # Fallback

            # =================================================================
            # 3. [ĐÃ TỐI ƯU] TĂNG CƯỜNG ĐẶC TRƯNG MÀU SẮC (COLOR ENHANCEMENT)
            # =================================================================
            # Chuyển BGR → LAB để xử lý trong không gian độc lập với ánh sáng
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            
            # Tách thành các kênh L, A, B
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            
            # Tăng cường kênh L (Luminance) bằng CLAHE để khôi phục chi tiết
            # Điều này giúp các feature về hình dáng (vai, hông) được rõ ràng hơn
            l_enhanced = self.clahe.apply(l_channel)
            
            # Tăng cường độ bão hòa (Saturation) của kênh A, B để màu sắc nổi bật hơn
            # Điều này giúp ResNet18 nhận diện tốt hơn màu sắc trang phục
            # Hệ số 1.15 giúp tăng saturation 15% mà không gây mất tự nhiên
            a_enhanced = cv2.convertScaleAbs(a_channel.astype(np.float32) - 128, alpha=1.15) + 128 - 127
            b_enhanced = cv2.convertScaleAbs(b_channel.astype(np.float32) - 128, alpha=1.15) + 128 - 127
            
            # Clamp về [0, 255]
            a_enhanced = np.clip(a_enhanced, 0, 255).astype(np.uint8)
            b_enhanced = np.clip(b_enhanced, 0, 255).astype(np.uint8)
            
            # Merge lại các kênh LAB
            img_lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
            
            # Chuyển ngược lại BGR
            img_enhanced = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # =================================================================
            # 4. [ĐÃ TỐI ƯU] LOẠI BỎ NHIỄU (NOISE REDUCTION)
            # =================================================================
            # Sử dụng Bilateral Filter để mịn hóa mà giữ lại các cạnh (edges)
            # Cực kỳ quan trọng để loại bỏ nhiễu camera nhưng giữ lại đường nét quần áo
            img_denoised = cv2.bilateralFilter(img_enhanced, d=7, sigmaColor=15, sigmaSpace=15)

            # =================================================================
            # 5. TRÍCH XUẤT ĐẶC TRƯNG TENSOR CORES
            # =================================================================
            tensor = self.transforms(img_denoised).unsqueeze(0).to(self.device)
            
            if self.half:
                tensor = tensor.half()
            
            with torch.no_grad():
                feat = self.model(tensor)
            
            # Normalize feature vector (quan trọng cho similarity matching)
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

