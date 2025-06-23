import numpy as np
import cv2 # For image manipulation if needed, e.g. cropping faces
import logging
from typing import List, Tuple, Dict, Any, Optional

# Assuming FaceDetector and FeatureExtractor are in the same directory or accessible via python path
from .detection import FaceDetector
from .embedding import FeatureExtractor

logger = logging.getLogger(__name__)

class RecognitionSystem:
    """
    Orchestrates face detection, feature extraction, and recognition.
    """
    def __init__(self,
                 detector: FaceDetector,
                 extractor: FeatureExtractor,
                 similarity_threshold: float = 0.5): # Default threshold, can be tuned
        """
        Initializes the recognition system.

        Args:
            detector (FaceDetector): An instance of the FaceDetector class.
            extractor (FeatureExtractor): An instance of the FeatureExtractor class.
            similarity_threshold (float): Minimum cosine similarity to consider a match.
        """
        self.detector = detector
        self.extractor = extractor
        self.similarity_threshold = similarity_threshold

        # In-memory database for registered faces: {name: [list of embedding vectors]}
        self.registered_faces: Dict[str, List[np.ndarray]] = {}
        logger.info(f"RecognitionSystem initialized with threshold: {self.similarity_threshold}")

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculates cosine similarity between two embedding vectors."""
        if emb1 is None or emb2 is None:
            return 0.0
        # Embeddings should already be L2 normalized by FeatureExtractor
        # dot_product = np.dot(emb1, emb2)
        # norm_emb1 = np.linalg.norm(emb1)
        # norm_emb2 = np.linalg.norm(emb2)
        # if norm_emb1 == 0 or norm_emb2 == 0: return 0.0
        # similarity = dot_product / (norm_emb1 * norm_emb2)

        # If embeddings are already L2 normalized, dot product is cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(similarity)

    def register_face(self, image: np.ndarray, name: str) -> bool:
        """
        Registers a face from an image.
        Detects the largest face, extracts its features, and stores them.

        Args:
            image (np.ndarray): The image containing the face (BGR format).
            name (str): The name associated with the face.

        Returns:
            bool: True if registration was successful, False otherwise.
        """
        if image is None or image.size == 0:
            logger.warning(f"Cannot register face for '{name}': Empty image provided.")
            return False
        if not name:
            logger.warning("Cannot register face: Name is empty.")
            return False

        detections = self.detector.detect_faces(image)
        if not detections:
            logger.warning(f"No faces detected in image for registering '{name}'.")
            return False

        # Select the largest face for registration (or the first one if sizes are equal)
        # Detections are [xmin, ymin, xmax, ymax, score]
        largest_detection = None
        max_area = 0
        for det_item in detections: # det_item is like [xmin, ymin, xmax, ymax, score]
            current_xmin, current_ymin, current_xmax, current_ymax = map(int, det_item[:4])
            area = (current_xmax - current_xmin) * (current_ymax - current_ymin)
            if area > max_area:
                max_area = area
                largest_detection = det_item

        if largest_detection is None: # Should not happen if detections is not empty
             logger.error("Internal error: detections found but no largest_detection selected.")
             return False

        # largest_detection is [xmin, ymin, xmax, ymax, score]
        # We only need coordinates for cropping here. Score is not used for registration logic itself.
        xmin, ymin, xmax, ymax = map(int, largest_detection[:4])

        # Crop the face patch
        # Add some padding or ensure crop is within image boundaries
        ih, iw = image.shape[:2]
        xmin_crop = max(0, xmin)
        ymin_crop = max(0, ymin)
        xmax_crop = min(iw, xmax)
        ymax_crop = min(ih, ymax)

        if xmax_crop <= xmin_crop or ymax_crop <= ymin_crop :
            logger.warning(f"Invalid crop dimensions for '{name}' after bounding box adjustment. Skipping registration.")
            return False

        face_patch = image[ymin_crop:ymax_crop, xmin_crop:xmax_crop]

        if face_patch.size == 0:
            logger.warning(f"Cropped face patch is empty for '{name}'. Skipping registration.")
            return False

        embedding = self.extractor.extract_features(face_patch)
        if embedding is None:
            logger.warning(f"Could not extract features for '{name}'. Registration failed.")
            return False

        if name not in self.registered_faces:
            self.registered_faces[name] = []
        self.registered_faces[name].append(embedding)
        logger.info(f"Successfully registered face for '{name}'. Total embeddings for this name: {len(self.registered_faces[name])}")
        return True

    def recognize_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognizes faces in an image.

        Args:
            image (np.ndarray): The image to process (BGR format).

        Returns:
            List[Dict[str, Any]]: A list of recognized faces, where each dict contains:
                'bbox': [xmin, ymin, xmax, ymax]
                'name': str (recognized name or 'Unknown')
                'score': float (confidence score of detection)
                'similarity': float (similarity to the matched registered face, if any)
        """
        if image is None or image.size == 0:
            logger.warning("Cannot recognize faces: Empty image provided.")
            return []

        detections = self.detector.detect_faces(image)
        recognized_results = []

        if not detections:
            return []

        ih, iw = image.shape[:2]

        for det in detections:
            xmin, ymin, xmax, ymax, score = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4]

            # Ensure crop is valid
            xmin_crop = max(0, xmin)
            ymin_crop = max(0, ymin)
            xmax_crop = min(iw, xmax)
            ymax_crop = min(ih, ymax)

            if xmax_crop <= xmin_crop or ymax_crop <= ymin_crop :
                logger.warning(f"Invalid crop dimensions for a detected face. Skipping.")
                face_result = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "name": "ErrorInCrop",
                    "score": score,
                    "similarity": 0.0
                }
                recognized_results.append(face_result)
                continue

            face_patch = image[ymin_crop:ymax_crop, xmin_crop:xmax_crop]

            if face_patch.size == 0:
                logger.warning("Cropped face patch is empty for a detected face. Skipping.")
                face_result = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "name": "EmptyCrop",
                    "score": score,
                    "similarity": 0.0
                }
                recognized_results.append(face_result)
                continue

            embedding = self.extractor.extract_features(face_patch)

            best_match_name = "Unknown"
            best_similarity = 0.0

            if embedding is not None:
                for name, known_embeddings in self.registered_faces.items():
                    for known_emb in known_embeddings:
                        similarity = self._cosine_similarity(embedding, known_emb)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            if similarity >= self.similarity_threshold:
                                best_match_name = name
                            # else, it's still unknown but we track best_similarity below threshold

            face_result = {
                "bbox": [xmin, ymin, xmax, ymax],
                "name": best_match_name if best_similarity >= self.similarity_threshold else "Unknown",
                "score": score, # Detection score
                "similarity": best_similarity # Similarity score
            }
            recognized_results.append(face_result)

            if best_match_name != "Unknown":
                logger.debug(f"Recognized {best_match_name} with similarity {best_similarity:.2f} (Detection score: {score:.2f})")
            elif embedding is not None :
                 logger.debug(f"Unknown face detected. Best similarity: {best_similarity:.2f} (Detection score: {score:.2f})")


        return recognized_results

    def set_similarity_threshold(self, threshold: float):
        """Updates the similarity threshold."""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            logger.info(f"Similarity threshold updated to: {self.similarity_threshold}")
        else:
            logger.warning(f"Invalid similarity threshold: {threshold}. Must be between 0.0 and 1.0.")

    def get_registered_faces_count(self) -> Dict[str, int]:
        """Returns a count of embeddings per registered name."""
        return {name: len(embeddings) for name, embeddings in self.registered_faces.items()}

# The __main__ block below was causing a SyntaxError.
# It has been removed as its functionality is covered by tests/test_core_components.py
# and it's not essential for the library part of this module.

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     logger.info("Starting RecognitionSystem Test...")
# ... (rest of the original __main__ block removed) ...
