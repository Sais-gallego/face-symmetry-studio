"""Face cloning, bilateral mirroring, and seamless seam blending."""

import cv2
import numpy as np
from typing import Optional, Tuple
from .detector import FaceLandmarks

class FaceCloner:
    def __init__(self, feather_radius: int = 12):
        self.feather_radius = feather_radius

    def _blend_seam(self, left_side: np.ndarray, right_side: np.ndarray) -> np.ndarray:
        """Blend the central vertical seam using a cosine transition mask to avoid hard creases."""
        h, wl = left_side.shape[:2]
        _, wr = right_side.shape[:2]
        
        # Raw joined image
        joined = np.hstack((left_side, right_side))
        
        # Feathering region around x = wl
        r = min(self.feather_radius, wl - 1, wr - 1)
        if r <= 2:
            return joined
            
        x_start = wl - r
        x_end = wl + r
        
        # Create smooth linear / cosine alpha ramp from 1.0 (left) to 0.0 (right)
        ramp = np.linspace(1.0, 0.0, x_end - x_start).reshape(1, -1, 1)
        
        # Blend left extended and right extended
        # In a symmetric face: left_side is mirrored, right_side is original or vice-versa
        # The transition region smooths out lighting / tone variances at the nose bridge
        left_strip = joined[:, x_start:x_end].astype(np.float32)
        flipped_strip = np.fliplr(left_strip)
        blended_strip = (left_strip * ramp + flipped_strip * (1.0 - ramp)).astype(np.uint8)
        
        joined[:, x_start:x_end] = blended_strip
        return joined

    def create_left_symmetric_face(self, image: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Generate a 100% left-symmetrical face (left hemisphere mirrored onto right)."""
        fx, fy, fw, fh = landmarks.face_box
        mid_x = landmarks.midline_x
        
        # Left half from midline to left edge of face
        left_half = image[fy:fy+fh, fx:mid_x]
        if left_half.shape[1] == 0 or left_half.shape[0] == 0:
            return image[fy:fy+fh, fx:fx+fw]
            
        # Mirror left half horizontally to create right half
        mirrored_right = cv2.flip(left_half, 1)
        
        # Blend central seam
        symmetric_face = self._blend_seam(left_half, mirrored_right)
        return cv2.resize(symmetric_face, (fw, fh))

    def create_right_symmetric_face(self, image: np.ndarray, landmarks: FaceLandmarks) -> np.ndarray:
        """Generate a 100% right-symmetrical face (right hemisphere mirrored onto left)."""
        fx, fy, fw, fh = landmarks.face_box
        mid_x = landmarks.midline_x
        
        # Right half from midline to right edge of face
        right_half = image[fy:fy+fh, mid_x:fx+fw]
        if right_half.shape[1] == 0 or right_half.shape[0] == 0:
            return image[fy:fy+fh, fx:fx+fw]
            
        # Mirror right half horizontally to create left half
        mirrored_left = cv2.flip(right_half, 1)
        
        # Blend central seam
        symmetric_face = self._blend_seam(mirrored_left, right_half)
        return cv2.resize(symmetric_face, (fw, fh))

    def clone_face_seamless(self, src_face: np.ndarray, target_canvas: np.ndarray, center_pt: Tuple[int, int]) -> np.ndarray:
        """Clone source face seamlessly into a target canvas using Poisson Gradient Blending."""
        h, w = src_face.shape[:2]
        
        # Elliptical mask for natural facial blending
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.44), int(h * 0.48))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Verify bounds on target
        cx, cy = center_pt
        th, tw = target_canvas.shape[:2]
        if cx - w // 2 < 0 or cx + w // 2 >= tw or cy - h // 2 < 0 or cy + h // 2 >= th:
            # Fallback simple overlay if boundary clipped
            return target_canvas
            
        cloned = cv2.seamlessClone(
            src_face,
            target_canvas,
            mask,
            center_pt,
            cv2.NORMAL_CLONE
        )
        return cloned
