"""Facial symmetry calculation and asymmetry heatmap analysis."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from .detector import FaceLandmarks

@dataclass
class SymmetryMetrics:
    overall_symmetry_score: float      # 0.0 - 100.0%
    pixel_similarity_score: float      # 0.0 - 100.0%
    ocular_symmetry_score: float       # 0.0 - 100.0%
    golden_ratio_score: float          # 0.0 - 100.0%
    asymmetry_heatmap: np.ndarray      # BGR heatmap image
    left_hemisphere_area: int
    right_hemisphere_area: int

class SymmetryAnalyzer:
    def __init__(self, golden_ratio: float = 1.61803398875):
        self.phi = golden_ratio

    def analyze_symmetry(self, image: np.ndarray, landmarks: FaceLandmarks) -> SymmetryMetrics:
        """Analyze bilateral facial symmetry and generate divergence heatmap."""
        fx, fy, fw, fh = landmarks.face_box
        mid_x = landmarks.midline_x
        
        # Calculate half width from midline
        half_w = min(mid_x - fx, (fx + fw) - mid_x)
        if half_w <= 10 or fh <= 10:
            # Fallback if box is degenerate
            half_w = fw // 2
            mid_x = fx + half_w
            
        left_crop = image[fy:fy+fh, mid_x - half_w:mid_x]
        right_crop = image[fy:fy+fh, mid_x:mid_x + half_w]
        
        # Mirror right crop to compare with left crop
        right_flipped = cv2.flip(right_crop, 1)
        
        # Resize right_flipped to match left_crop exactly if 1-pixel discrepancy
        if left_crop.shape != right_flipped.shape:
            right_flipped = cv2.resize(right_flipped, (left_crop.shape[1], left_crop.shape[0]))
            
        # Convert to grayscale for structural dissimilarity
        gray_left = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY) if len(left_crop.shape) == 3 else left_crop
        gray_right = cv2.cvtColor(right_flipped, cv2.COLOR_BGR2GRAY) if len(right_flipped.shape) == 3 else right_flipped
        
        # Compute absolute pixel difference
        abs_diff = cv2.absdiff(gray_left, gray_right)
        
        # Normalization and pixel similarity score (100% is identical)
        mean_diff = float(np.mean(abs_diff))
        pixel_similarity = max(0.0, min(100.0, 100.0 - (mean_diff / 255.0) * 100.0 * 2.0))
        
        # Ocular symmetry (pupil distance to midline balance)
        ocular_score = 100.0
        if landmarks.left_eye and landmarks.right_eye:
            d_left = abs(landmarks.left_eye[0] - mid_x)
            d_right = abs(landmarks.right_eye[0] - mid_x)
            diff_ratio = abs(d_left - d_right) / max(1.0, float(max(d_left, d_right)))
            ocular_score = max(0.0, min(100.0, (1.0 - diff_ratio) * 100.0))
            
        # Golden ratio estimation (face height to face width vs Phi)
        ratio = float(fh) / max(1.0, float(fw))
        ratio_error = abs(ratio - self.phi) / self.phi
        golden_score = max(0.0, min(100.0, (1.0 - ratio_error) * 100.0))
        
        # Weighted overall symmetry index
        overall = (pixel_similarity * 0.50) + (ocular_score * 0.35) + (golden_score * 0.15)
        overall = round(max(0.0, min(100.0, overall)), 2)
        
        # Build bilateral Asymmetry Heatmap
        # Smooth high-frequency noise with Gaussian blur
        smoothed_diff = cv2.GaussianBlur(abs_diff, (15, 15), 0)
        norm_diff = cv2.normalize(smoothed_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Colorize half heatmap using INFERNO or JET
        half_heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        
        # Create full face heatmap by mirroring
        flipped_half_heatmap = cv2.flip(half_heatmap, 1)
        full_heatmap = np.hstack((half_heatmap, flipped_half_heatmap))
        
        # Resize to full face size
        full_heatmap = cv2.resize(full_heatmap, (fw, fh))
        
        # Blend heatmap with original face for anatomical context
        face_original = image[fy:fy+fh, fx:fx+fw]
        if face_original.shape[:2] != full_heatmap.shape[:2]:
            full_heatmap = cv2.resize(full_heatmap, (face_original.shape[1], face_original.shape[0]))
            
        blended_heatmap = cv2.addWeighted(face_original, 0.45, full_heatmap, 0.55, 0)
        
        return SymmetryMetrics(
            overall_symmetry_score=overall,
            pixel_similarity_score=round(pixel_similarity, 2),
            ocular_symmetry_score=round(ocular_score, 2),
            golden_ratio_score=round(golden_score, 2),
            asymmetry_heatmap=blended_heatmap,
            left_hemisphere_area=int(left_crop.shape[0] * left_crop.shape[1]),
            right_hemisphere_area=int(right_crop.shape[0] * right_crop.shape[1])
        )
