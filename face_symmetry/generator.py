"""Multi-panel visual studio and diagnostic collage generator."""

import cv2
import numpy as np
from typing import Optional
from .detector import FaceDetector, FaceLandmarks
from .symmetry import SymmetryAnalyzer, SymmetryMetrics
from .cloner import FaceCloner

class SymmetryStudio:
    def __init__(self, face_cascade: Optional[str] = None, eye_cascade: Optional[str] = None):
        self.detector = FaceDetector(face_cascade, eye_cascade)
        self.analyzer = SymmetryAnalyzer()
        self.cloner = FaceCloner()

    def process_portrait(self, image: np.ndarray, target_size: int = 400) -> dict:
        """Run full biometric symmetry pipeline on an image."""
        aligned_img, landmarks = self.detector.align_face(image)
        metrics = self.analyzer.analyze_symmetry(aligned_img, landmarks)
        
        fx, fy, fw, fh = landmarks.face_box
        original_face = aligned_img[fy:fy+fh, fx:fx+fw]
        original_face = cv2.resize(original_face, (target_size, target_size))
        
        # Overlay midline and eyes on a copy of original
        annotated_orig = original_face.copy()
        scale_x = target_size / max(1, fw)
        scale_y = target_size / max(1, fh)
        
        rel_mid_x = int((landmarks.midline_x - fx) * scale_x)
        cv2.line(annotated_orig, (rel_mid_x, 0), (rel_mid_x, target_size), (255, 255, 0), 2)
        
        if landmarks.left_eye:
            lex = int((landmarks.left_eye[0] - fx) * scale_x)
            ley = int((landmarks.left_eye[1] - fy) * scale_y)
            cv2.circle(annotated_orig, (lex, ley), 5, (0, 255, 0), -1)
        if landmarks.right_eye:
            rex = int((landmarks.right_eye[0] - fx) * scale_x)
            rey = int((landmarks.right_eye[1] - fy) * scale_y)
            cv2.circle(annotated_orig, (rex, rey), 5, (0, 255, 0), -1)

        # Generate left-left and right-right faces
        left_sym = self.cloner.create_left_symmetric_face(aligned_img, landmarks)
        left_sym = cv2.resize(left_sym, (target_size, target_size))
        
        right_sym = self.cloner.create_right_symmetric_face(aligned_img, landmarks)
        right_sym = cv2.resize(right_sym, (target_size, target_size))
        
        heatmap = cv2.resize(metrics.asymmetry_heatmap, (target_size, target_size))
        
        # Build 4-panel collage
        collage = self._create_collage(
            annotated_orig,
            left_sym,
            heatmap,
            right_sym,
            metrics,
            panel_size=target_size
        )
        
        return {
            "metrics": metrics,
            "landmarks": landmarks,
            "original_face": original_face,
            "annotated_original": annotated_orig,
            "left_symmetric": left_sym,
            "right_symmetric": right_sym,
            "heatmap": heatmap,
            "collage": collage
        }

    def _create_collage(
        self,
        orig: np.ndarray,
        left_sym: np.ndarray,
        heatmap: np.ndarray,
        right_sym: np.ndarray,
        metrics: SymmetryMetrics,
        panel_size: int = 400
    ) -> np.ndarray:
        """Compose 4 panels with headers and metric footer into a single image."""
        header_h = 45
        footer_h = 70
        total_w = panel_size * 2 + 30
        total_h = (panel_size + header_h) * 2 + footer_h + 30
        
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 27)  # Dark slate background (#18181b)
        
        def add_panel(img: np.ndarray, title: str, top_left_x: int, top_left_y: int):
            # Title bar
            cv2.putText(
                canvas, title, (top_left_x + 10, top_left_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA
            )
            # Image
            canvas[top_left_y + header_h : top_left_y + header_h + panel_size, top_left_x : top_left_x + panel_size] = img
            # Border
            cv2.rectangle(
                canvas,
                (top_left_x, top_left_y + header_h),
                (top_left_x + panel_size, top_left_y + header_h + panel_size),
                (60, 60, 70), 1
            )

        # Panel 1: Original + Midline
        add_panel(orig, "1. ORIGINAL (Sagittal Midline)", 10, 10)
        # Panel 2: Left-Left Symmetric Clone
        add_panel(left_sym, "2. LEFT-LEFT SYMMETRIC", panel_size + 20, 10)
        # Panel 3: Asymmetry Heatmap
        add_panel(heatmap, "3. ASYMMETRY DIVERGENCE HEATMAP", 10, panel_size + header_h + 20)
        # Panel 4: Right-Right Symmetric Clone
        add_panel(right_sym, "4. RIGHT-RIGHT SYMMETRIC", panel_size + 20, panel_size + header_h + 20)
        
        # Footer metrics bar
        footer_y = total_h - footer_h + 5
        score_text = f"OVERALL SYMMETRY INDEX: {metrics.overall_symmetry_score}%"
        sub_text = f"Pixel Sim: {metrics.pixel_similarity_score}%  |  Ocular Balance: {metrics.ocular_symmetry_score}%  |  Phi Ratio: {metrics.golden_ratio_score}%"
        
        cv2.putText(canvas, score_text, (20, footer_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 115), 2, cv2.LINE_AA)
        cv2.putText(canvas, sub_text, (20, footer_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 190), 1, cv2.LINE_AA)
        
        return canvas
