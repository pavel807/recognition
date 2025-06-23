import tensorflow as tf
import numpy as np
import cv2
import logging
import os

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Handles face feature extraction (embedding generation) using a TensorFlow Lite model
    (e.g., MobileFaceNet).
    """
    def __init__(self, model_path='src/models/mobilefacenet.tflite'):
        """
        Initializes the feature extractor.

        Args:
            model_path (str): Path to the TFLite face embedding model.
        """
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            logger.error(f"Embedding model file not found: {self.model_path}")
            raise FileNotFoundError(f"Embedding model file not found: {self.model_path}")

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()

    def _load_model(self):
        """Loads the TFLite model and allocates tensors."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(f"Feature extraction model loaded successfully: {self.model_path}")
            logger.info(f"Input details: {self.input_details}")
            logger.info(f"Output details: {self.output_details}")
        except Exception as e:
            logger.error(f"Failed to load TFLite embedding model: {e}")
            raise RuntimeError(f"Failed to load TFLite embedding model: {e}")

    def _preprocess_face_patch(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocesses a cropped face image for the embedding model.
        Common for MobileFaceNet: 112x112, RGB, normalized.
        Normalization might be (X - 127.5) / 128.0 or similar.
        """
        # Assuming input tensor is [1, height, width, channels]
        input_shape = self.input_details[0]['shape']
        target_height, target_width = input_shape[1], input_shape[2] # e.g., 112, 112

        if face_image.shape[0] == 0 or face_image.shape[1] == 0:
            logger.error("Received empty face patch for preprocessing.")
            # Return a zero array of the correct type and shape if you want to handle this gracefully
            # Or raise an error. For now, let's assume this won't happen with proper upstream checks.
            # This would cause cv2.resize to fail.
            raise ValueError("Empty face patch received in _preprocess_face_patch")


        img_resized = cv2.resize(face_image, (target_width, target_height))

        if img_resized.shape[2] == 3 and self.input_details[0]['dtype'] == np.float32: # Check if BGR
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else: # Already RGB or single channel (less likely for face models)
            img_rgb = img_resized

        img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0 # Common normalization for MobileFaceNet

        # Add batch dimension
        return np.expand_dims(img_normalized, axis=0)

    def extract_features(self, face_image: np.ndarray) -> np.ndarray | None:
        """
        Extracts features (embedding) from a cropped face image.

        Args:
            face_image (np.ndarray): The cropped face image (BGR format).
                                     It's assumed this image is already cropped
                                     around a detected face.

        Returns:
            np.ndarray: The feature vector (embedding), or None if an error occurs.
        """
        if self.interpreter is None:
            logger.error("Embedding model not loaded. Cannot extract features.")
            return None

        if face_image is None or face_image.size == 0:
            logger.warning("Cannot extract features from an empty or None image.")
            return None

        try:
            input_data = self._preprocess_face_patch(face_image)
        except ValueError as e:
            logger.error(f"Error during preprocessing of face patch: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during preprocessing: {e}")
            return None


        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            features = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Output is typically [1, embedding_size], e.g., [1, 128] or [1, 512]
            # Normalize the embedding vector (L2 normalization) - common practice
            features_normalized = features[0] / np.linalg.norm(features[0])
            return features_normalized

        except Exception as e:
            logger.error(f"Error during feature extraction inference: {e}")
            return None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create a dummy face patch for testing (e.g., 100x100 BGR)
    # This should ideally be similar to what a face detector would crop.
    dummy_face_patch = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # IMPORTANT: This test will likely FAIL or produce non-meaningful embeddings
    # if the placeholder mobilefacenet.tflite is not replaced with an actual model file.
    logger.warning("Using placeholder model file for feature extractor. Results will not be meaningful.")

    # Create dummy model file if it doesn't exist
    model_file_path = 'src/models/mobilefacenet.tflite'
    if not os.path.exists(model_file_path):
        with open(model_file_path, 'w') as f: f.write("dummy tflite content")
        logger.info(f"Created dummy model file: {model_file_path}")

    try:
        extractor = FeatureExtractor(model_path=model_file_path)

        logger.info("Attempting to extract features with placeholder model...")
        # The input to extract_features should be a BGR image patch of a face
        features = extractor.extract_features(dummy_face_patch)

        if features is not None:
            logger.info(f"Extracted 'features' (using placeholder model). Shape: {features.shape}, Type: {features.dtype}")
            # logger.info(f"First few values: {features[:5]}") # Print some values
        else:
            logger.info("Failed to extract 'features' (as expected with placeholder or if model/preprocessing is incompatible).")

    except FileNotFoundError as e:
        logger.error(f"Test failed: {e}. Ensure model file exists or is correctly pathed.")
    except RuntimeError as e:
        logger.error(f"Test failed during model loading or inference: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during testing: {e}")
    finally:
        # Clean up dummy model file if created by this test
        # For now, managed outside or replaced by actual files.
        pass
