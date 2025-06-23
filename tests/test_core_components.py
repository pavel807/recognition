import unittest
import numpy as np
import os
import sys

# Add src directory to Python path to allow direct import of modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.recognition import RecognitionSystem
from src.core.detection import FaceDetector
from src.core.embedding import FeatureExtractor

# Helper to create dummy model files if they don't exist, to allow tests to run
# These paths should point to the *source* location of models for py2app to find them.
# Assuming this test script is in project_root/tests/
project_root_for_tests = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_src_dir_for_tests = os.path.join(project_root_for_tests, 'src', 'models')

blazeface_model_path_for_tests = os.path.join(models_src_dir_for_tests, 'blazeface_frontend.tflite')
anchors_path_for_tests = os.path.join(models_src_dir_for_tests, 'anchors.npy')
mobilefacenet_model_path_for_tests = os.path.join(models_src_dir_for_tests, 'mobilefacenet.tflite')

def ensure_dummy_models_exist_for_tests():
    os.makedirs(models_src_dir_for_tests, exist_ok=True)

    # Delete existing text placeholders first to ensure binary versions are used
    # These paths point to src/models/ where the main app also looks in dev.
    if os.path.exists(blazeface_model_path_for_tests):
        try:
            os.remove(blazeface_model_path_for_tests)
        except OSError: pass
    if os.path.exists(mobilefacenet_model_path_for_tests):
        try:
            os.remove(mobilefacenet_model_path_for_tests)
        except OSError: pass

    # Create minimal binary dummy models
    # Using a slightly more extended header to avoid other potential immediate load errors.
    # Content: TFL3 + version (4 bytes) + offset to schema (4 bytes) + schema_len (4 bytes) + buffer_offset (4 bytes)
    # This is still not a valid model but might pass more initial checks.
    minimal_tflite_header = b'TFL3\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00'
    with open(blazeface_model_path_for_tests, 'wb') as f: f.write(minimal_tflite_header)
    with open(mobilefacenet_model_path_for_tests, 'wb') as f: f.write(minimal_tflite_header)

    # anchors.npy is created correctly if it doesn't exist.
    if not os.path.exists(anchors_path_for_tests):
        dummy_anchors = np.random.rand(896, 4).astype(np.float32)
        np.save(anchors_path_for_tests, dummy_anchors)

class TestCoreComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ensure_dummy_models_exist_for_tests()

        class MockFaceDetector:
            def __init__(self, model_path=None, anchors_path=None): # Match signature
                pass
            def detect_faces(self, image: np.ndarray) -> list:
                h, w = image.shape[:2]
                if np.any(image):
                    return [[w//4, h//4, 3*w//4, 3*h//4, 0.95]]
                return []

        class MockFeatureExtractor:
            def __init__(self, model_path=None):
                pass
            def extract_features(self, face_image: np.ndarray) -> np.ndarray | None:
                if face_image is None or face_image.size == 0: return None

                # PersonA: Blue channel prominent
                if np.mean(face_image[:,:,0]) > 50 :
                    vec = np.array([0.1] * 128, dtype=np.float32)
                    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
                # Unknown (for test): Green channel prominent
                elif np.mean(face_image[:,:,1]) > 50:
                    # Create a vector known to be different from PersonA's
                    unknown_vec_components = [0.0] * 128
                    unknown_vec_components[0] = 0.0 # Ensure it's different from PersonA's [0.1, 0.1,...]
                    unknown_vec_components[1] = 0.2 # e.g. make it somewhat orthogonal or just distinct
                    vec = np.array(unknown_vec_components, dtype=np.float32)
                    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
                # Fallback "other unknown" if needed for other tests, or generic random
                else:
                    # For any other image not matching above, provide another distinct vector
                    # This helps ensure test_registration_and_recognition_mocked only matches PersonA
                    # when PersonA's image is provided.
                    other_vec_components = [0.0] * 128
                    other_vec_components[0] = 0.5 # Distinct from PersonA and Green-Unknown
                    vec = np.array(other_vec_components, dtype=np.float32)
                    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec

        cls.mock_detector = MockFaceDetector()
        cls.mock_extractor = MockFeatureExtractor()
        cls.recognition_system = RecognitionSystem(
            detector=cls.mock_detector,
            extractor=cls.mock_extractor,
            similarity_threshold=0.8
        )

    def test_cosine_similarity(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        vec4 = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(self.recognition_system._cosine_similarity(vec1, vec2), 1.0)
        self.assertAlmostEqual(self.recognition_system._cosine_similarity(vec1, vec3), 0.0)
        self.assertAlmostEqual(self.recognition_system._cosine_similarity(vec1, vec4), -1.0)
        self.assertAlmostEqual(self.recognition_system._cosine_similarity(vec1, None), 0.0)

    def test_model_loading_placeholders(self):
        try:
            # These paths now point to src/models directly, which is where dummy files are ensured
            detector = FaceDetector(model_path=blazeface_model_path_for_tests, anchors_path=anchors_path_for_tests)
            self.assertIsNotNone(detector.interpreter, "FaceDetector interpreter should be loaded.")
        except Exception as e:
            self.fail(f"FaceDetector instantiation failed with placeholder model: {e}")

        try:
            extractor = FeatureExtractor(model_path=mobilefacenet_model_path_for_tests)
            self.assertIsNotNone(extractor.interpreter, "FeatureExtractor interpreter should be loaded.")
        except Exception as e:
            self.fail(f"FeatureExtractor instantiation failed with placeholder model: {e}")

    def test_registration_and_recognition_mocked(self):
        dummy_image_person_a = np.zeros((200, 200, 3), dtype=np.uint8)
        dummy_image_person_a[50:150, 50:150, 0] = 100
        reg_success = self.recognition_system.register_face(dummy_image_person_a, "TestPersonA")
        self.assertTrue(reg_success, "Registration should succeed.")
        self.assertIn("TestPersonA", self.recognition_system.registered_faces)

        results_a = self.recognition_system.recognize_faces(dummy_image_person_a)
        self.assertEqual(len(results_a), 1)
        self.assertEqual(results_a[0]['name'], "TestPersonA")

        dummy_image_unknown = np.zeros((200, 200, 3), dtype=np.uint8)
        dummy_image_unknown[50:150, 50:150, 1] = 100
        results_unknown = self.recognition_system.recognize_faces(dummy_image_unknown)
        self.assertEqual(len(results_unknown), 1)
        self.assertEqual(results_unknown[0]['name'], "Unknown")

    def test_empty_image_registration(self):
        empty_image = np.array([])
        reg_success = self.recognition_system.register_face(empty_image, "EmptyTest")
        self.assertFalse(reg_success)

        empty_image_2 = np.zeros((0,0,3), dtype=np.uint8)
        reg_success_2 = self.recognition_system.register_face(empty_image_2, "EmptyTest2")
        self.assertFalse(reg_success_2)

    def test_no_face_detected_registration(self):
        all_black_image = np.zeros((100,100,3), dtype=np.uint8)
        reg_success = self.recognition_system.register_face(all_black_image, "NoFaceTest")
        self.assertFalse(reg_success)

    def test_recognition_no_faces_detected(self):
        all_black_image = np.zeros((100,100,3), dtype=np.uint8)
        results = self.recognition_system.recognize_faces(all_black_image)
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
