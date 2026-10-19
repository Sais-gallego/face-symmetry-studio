import unittest
import numpy as np
import cv2
from face_symmetry.detector import FaceDetector, FaceLandmarks
from face_symmetry.symmetry import SymmetryAnalyzer, SymmetryMetrics
from face_symmetry.cloner import FaceCloner
from face_symmetry.generator import SymmetryStudio
from face_symmetry.cli import create_synthetic_portrait
from face_symmetry.app import app

class TestFaceSymmetryStudio(unittest.TestCase):
    def setUp(self):
        self.portrait = create_synthetic_portrait(400, 500)
        self.studio = SymmetryStudio()

    def test_synthetic_portrait_generation(self):
        self.assertEqual(self.portrait.shape, (500, 400, 3))
        self.assertEqual(self.portrait.dtype, np.uint8)

    def test_face_detector_and_alignment(self):
        detector = FaceDetector()
        aligned, landmarks = detector.align_face(self.portrait)
        self.assertIsNotNone(aligned)
        self.assertIsInstance(landmarks, FaceLandmarks)
        self.assertGreater(landmarks.midline_x, 0)
        self.assertEqual(len(landmarks.face_box), 4)

    def test_symmetry_analyzer_metrics(self):
        analyzer = SymmetryAnalyzer()
        detector = FaceDetector()
        _, landmarks = detector.align_face(self.portrait)
        metrics = analyzer.analyze_symmetry(self.portrait, landmarks)
        
        self.assertIsInstance(metrics, SymmetryMetrics)
        self.assertGreaterEqual(metrics.overall_symmetry_score, 0.0)
        self.assertLessEqual(metrics.overall_symmetry_score, 100.0)
        self.assertGreaterEqual(metrics.pixel_similarity_score, 0.0)
        self.assertLessEqual(metrics.pixel_similarity_score, 100.0)
        self.assertEqual(metrics.asymmetry_heatmap.shape[:2], (landmarks.face_box[3], landmarks.face_box[2]))

    def test_left_and_right_cloner(self):
        cloner = FaceCloner()
        detector = FaceDetector()
        _, landmarks = detector.align_face(self.portrait)
        
        left_sym = cloner.create_left_symmetric_face(self.portrait, landmarks)
        right_sym = cloner.create_right_symmetric_face(self.portrait, landmarks)
        
        self.assertEqual(left_sym.shape[:2], (landmarks.face_box[3], landmarks.face_box[2]))
        self.assertEqual(right_sym.shape[:2], (landmarks.face_box[3], landmarks.face_box[2]))

    def test_full_studio_processing_and_collage(self):
        results = self.studio.process_portrait(self.portrait, target_size=300)
        self.assertIn("metrics", results)
        self.assertIn("collage", results)
        self.assertIn("left_symmetric", results)
        self.assertIn("right_symmetric", results)
        self.assertIn("heatmap", results)
        
        collage = results["collage"]
        self.assertEqual(len(collage.shape), 3)
        self.assertGreater(collage.shape[0], 600)
        self.assertGreater(collage.shape[1], 600)

    def test_flask_api_demo_endpoint(self):
        client = app.test_client()
        response = client.get("/api/demo")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("metrics", data)
        self.assertIn("collage", data)
        self.assertIn("left_symmetric", data)
        self.assertIn("right_symmetric", data)
        self.assertIn("heatmap", data)
        self.assertGreater(data["metrics"]["overall_symmetry_score"], 0)

if __name__ == "__main__":
    unittest.main()
