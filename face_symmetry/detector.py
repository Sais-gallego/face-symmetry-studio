"""Face detection, landmark estimation, and facial alignment module."""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FACE_CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_eye.xml")

@dataclass
class FaceLandmarks:
    face_box: Tuple[int, int, int, int]  # (x, y, w, h)
    left_eye: Optional[Tuple[int, int]] = None  # (x, y)
    right_eye: Optional[Tuple[int, int]] = None # (x, y)
    midline_x: int = 0
    roll_angle: float = 0.0

class FaceDetector:
    def __init__(self, face_cascade_path: Optional[str] = None, eye_cascade_path: Optional[str] = None):
        fpath = face_cascade_path or FACE_CASCADE_PATH
        epath = eye_cascade_path or EYE_CASCADE_PATH
        
        self.face_cascade = None
        self.eye_cascade = None
        
        # Check if OpenCV runtime supports CascadeClassifier
        if hasattr(cv2, "CascadeClassifier"):
            if os.path.exists(fpath):
                self.face_cascade = cv2.CascadeClassifier(fpath)
            if os.path.exists(epath):
                self.eye_cascade = cv2.CascadeClassifier(epath)

    def detect_primary_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the largest frontal face or return normalized bounding box."""
        if image is None or image.size == 0:
            return None
            
        h_img, w_img = image.shape[:2]
        
        if self.face_cascade is not None and not self.face_cascade.empty():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                return tuple(int(v) for v in faces[0])
                
        # Smart geometric fallback: center 70% of image with proper margins
        margin_x = int(w_img * 0.15)
        margin_y = int(h_img * 0.12)
        fw = w_img - 2 * margin_x
        fh = h_img - 2 * margin_y
        return (margin_x, margin_y, fw, fh)

    def detect_eyes(self, face_roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect eyes inside the upper half of the face ROI."""
        h, w = face_roi.shape[:2]
        
        if self.eye_cascade is not None and not self.eye_cascade.empty():
            upper_half = face_roi[0:int(h * 0.65), :]
            gray = cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY) if len(upper_half.shape) == 3 else upper_half
            eyes = self.eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 20)
            )
            if len(eyes) >= 2:
                return [tuple(int(v) for v in e) for e in eyes]

        # Anthropometric ocular estimation fallback (eyes typically at 34% and 66% width, 38% height)
        eye_y = int(h * 0.38)
        eye_w = int(w * 0.15)
        eye_h = int(h * 0.12)
        eye_l_x = int(w * 0.32) - eye_w // 2
        eye_r_x = int(w * 0.68) - eye_w // 2
        return [
            (eye_l_x, eye_y - eye_h // 2, eye_w, eye_h),
            (eye_r_x, eye_y - eye_h // 2, eye_w, eye_h)
        ]

    def align_face(self, image: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, FaceLandmarks]:
        """Align the face horizontally based on ocular axis and crop region."""
        h_img, w_img = image.shape[:2]
        
        if face_box is None:
            face_box = self.detect_primary_face(image)
                
        fx, fy, fw, fh = face_box
        face_roi = image[fy:fy+fh, fx:fx+fw]
        
        # Detect eyes
        eyes = self.detect_eyes(face_roi)
        left_eye, right_eye = None, None
        roll_angle = 0.0
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            e1, e2 = eyes[0], eyes[-1]
            
            p1 = (fx + e1[0] + e1[2] // 2, fy + e1[1] + e1[3] // 2)
            p2 = (fx + e2[0] + e2[2] // 2, fy + e2[1] + e2[3] // 2)
            
            left_eye, right_eye = p1, p2
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            if dx != 0:
                roll_angle = float(np.degrees(np.arctan2(dy, dx)))
        
        # If face is tilted by more than 1 degree, rotate image to align horizontally
        aligned_image = image.copy()
        if abs(roll_angle) > 1.0 and abs(roll_angle) < 45.0:
            center = (fx + fw // 2, fy + fh // 2)
            rot_mat = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
            aligned_image = cv2.warpAffine(
                image, rot_mat, (w_img, h_img),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT
            )
            new_box = self.detect_primary_face(aligned_image)
            if new_box:
                face_box = new_box
                fx, fy, fw, fh = face_box

        # Compute vertical midline sagittal axis
        if left_eye and right_eye:
            midline_x = (left_eye[0] + right_eye[0]) // 2
        else:
            midline_x = fx + fw // 2
            
        landmarks = FaceLandmarks(
            face_box=face_box,
            left_eye=left_eye,
            right_eye=right_eye,
            midline_x=midline_x,
            roll_angle=roll_angle
        )
        return aligned_image, landmarks
