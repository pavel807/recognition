import tensorflow as tf
import numpy as np
import cv2
import logging
import os

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Handles face detection using a TensorFlow Lite model (e.g., BlazeFace).
    """
    def __init__(self,
                 model_path='src/models/blazeface_frontend.tflite',
                 anchors_path='src/models/anchors.npy',
                 score_threshold=0.7,
                 iou_threshold=0.3):
        """
        Initializes the face detector.

        Args:
            model_path (str): Path to the TFLite face detection model.
            anchors_path (str): Path to the anchors numpy file (for BlazeFace).
            score_threshold (float): Threshold for detection scores.
            iou_threshold (float): Threshold for NMS IoU.
        """
        self.model_path = model_path
        self.anchors_path = anchors_path
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.anchors_path) and "blazeface" in self.model_path.lower():
             logger.warning(f"Anchors file not found: {self.anchors_path}. BlazeFace might not work correctly.")
             self.anchors = None # Or handle as critical error
        else:
            try:
                self.anchors = np.load(self.anchors_path)
            except Exception as e:
                logger.warning(f"Could not load anchors from {self.anchors_path} (assuming it's not a numpy file for placeholder): {e}")
                self.anchors = None # Placeholder, real anchors are crucial for blazeface

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
            logger.info(f"Face detection model loaded successfully: {self.model_path}")
            logger.info(f"Input details: {self.input_details}")
            logger.info(f"Output details: {self.output_details}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            # In a real app, might want to raise this or handle more gracefully
            raise RuntimeError(f"Failed to load TFLite model: {e}")


    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image for the model.
        Assumes BlazeFace-like input: 128x128, RGB, normalized to [-1, 1] or [0,1]
        depending on model specifics. This is a common configuration.
        """
        # Assuming input tensor is [1, height, width, channels]
        input_shape = self.input_details[0]['shape']
        target_height, target_width = input_shape[1], input_shape[2]

        img_resized = cv2.resize(image, (target_width, target_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize the image (example: to [-1, 1] if model expects that)
        # The exact normalization depends on the model's training
        # For BlazeFace, it's often division by 127.5 and subtraction of 1.0
        img_normalized = (img_rgb.astype(np.float32) / 127.5) - 1.0

        # Add batch dimension
        return np.expand_dims(img_normalized, axis=0)

    def _decode_predictions(self, raw_boxes, raw_scores, original_image_shape):
        """
        Decodes raw model predictions into bounding boxes.
        This is a placeholder for BlazeFace-specific decoding logic which involves anchors.
        A full implementation would use self.anchors.
        """
        detections = []
        img_h, img_w = original_image_shape[:2]

        # Assuming raw_scores is [1, num_detections, num_classes_or_1]
        # Assuming raw_boxes is [1, num_detections, 4] (y_center, x_center, h, w relative to anchors)

        # Simplified placeholder logic:
        # This part is highly dependent on the specific model's output format.
        # For BlazeFace, you'd iterate through scores and apply them to anchor boxes,
        # then convert anchor-relative coordinates to absolute image coordinates.

        # Example: if raw_scores are actual confidences and raw_boxes are already [ymin, xmin, ymax, xmax] normalized
        # This is NOT how BlazeFace typically works, but serves as a structural placeholder.

        num_detections = raw_scores.shape[1] # Or derive from model output spec

        for i in range(num_detections):
            score = raw_scores[0, i, 0] # Assuming single class score
            if score > self.score_threshold:
                # This decoding needs to be accurate for the chosen model
                # For BlazeFace, this involves complex calculations with anchors.
                # The box coordinates from BlazeFace are typically [y_center, x_center, h, w]
                # relative to pre-defined anchors.
                # Let's assume for this placeholder that raw_boxes are [ymin, xmin, ymax, xmax] normalized to [0,1]
                # THIS IS A GROSS SIMPLIFICATION AND WILL NOT WORK WITH A REAL BLAZEFACE MODEL
                # WITHOUT THE CORRECT ANCHOR-BASED DECODING.
                if self.anchors is not None and "blazeface" in self.model_path.lower() :
                    # If this were real BlazeFace, you'd use self.anchors here.
                    # logger.warning("Using placeholder decoding for BlazeFace. Real decoding with anchors is complex.")
                    # The following is just to make it runnable, not correct for BlazeFace
                    y_center, x_center, h, w = raw_boxes[0, i, 0], raw_boxes[0, i, 1], raw_boxes[0, i, 2], raw_boxes[0, i, 3]

                    # Example of applying to anchors (conceptually)
                    # anchor_y, anchor_x, anchor_h, anchor_w = self.anchors[i, :]
                    # box_y_center = y_center * anchor_h + anchor_y
                    # box_x_center = x_center * anchor_w + anchor_x
                    # box_h = np.exp(h) * anchor_h
                    # box_w = np.exp(w) * anchor_w

                    # ymin = (box_y_center - box_h / 2) * img_h
                    # xmin = (box_x_center - box_w / 2) * img_w
                    # ymax = (box_y_center + box_h / 2) * img_h
                    # xmax = (box_x_center + box_w / 2) * img_w

                    # Simplified box for placeholder if anchors are missing or logic incomplete
                    # This assumes raw_boxes are somewhat direct [ymin, xmin, ymax, xmax] normalized.
                    # THIS IS NOT BLAZEFACE STANDARD.
                    ymin = raw_boxes[0, i, 0] * img_h
                    xmin = raw_boxes[0, i, 1] * img_w
                    ymax = raw_boxes[0, i, 2] * img_h
                    xmax = raw_boxes[0, i, 3] * img_w

                else: # Generic placeholder if not blazeface or anchors missing
                    ymin, xmin, ymax, xmax = raw_boxes[0,i,0]*img_h, raw_boxes[0,i,1]*img_w, raw_boxes[0,i,2]*img_h, raw_boxes[0,i,3]*img_w

                detections.append([int(xmin), int(ymin), int(xmax), int(ymax), score])

        return np.array(detections)

    def _non_maximum_suppression(self, boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
        """
        Performs Non-Maximum Suppression.
        boxes: [[xmin, ymin, xmax, ymax, score], ...]
        """
        if len(boxes) == 0:
            return []

        # Sort by score
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        picked_boxes = []

        while boxes:
            current_box = boxes.pop(0)
            picked_boxes.append(current_box)

            remaining_boxes = []
            for box in boxes:
                # Calculate IoU
                x1_inter = max(current_box[0], box[0])
                y1_inter = max(current_box[1], box[1])
                x2_inter = min(current_box[2], box[2])
                y2_inter = min(current_box[3], box[3])

                inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
                current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                box_area = (box[2] - box[0]) * (box[3] - box[1])

                iou = inter_area / float(current_area + box_area - inter_area)

                if iou < iou_threshold:
                    remaining_boxes.append(box)
            boxes = remaining_boxes

        return np.array(picked_boxes)


    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detects faces in an image.

        Args:
            image (np.ndarray): The input image (BGR format).

        Returns:
            list: A list of detections, where each detection is
                  [xmin, ymin, xmax, ymax, score].
                  Returns an empty list if no faces are detected or an error occurs.
        """
        if self.interpreter is None:
            logger.error("Model not loaded. Cannot detect faces.")
            return []

        original_shape = image.shape
        input_data = self._preprocess_image(image)

        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Output details vary by model. For BlazeFace, typically:
            # output_details[0]: scores (e.g., shape [1, 896, 1])
            # output_details[1]: boxes (e.g., shape [1, 896, 4] - relative to anchors)
            # This order can differ! Always check your model's specifics.
            # Assuming output_details[0] is scores and output_details[1] is boxes for this example.
            # It's CRITICAL to map these correctly.

            # Find score and box tensors from output_details
            # This is more robust than assuming fixed indices
            scores_tensor_index = -1
            boxes_tensor_index = -1

            # A common setup for BlazeFace has one output for scores and one for boxes.
            # Scores are often [1, num_anchors, 1] and boxes [1, num_anchors, 4]
            # Or sometimes they are concatenated into one tensor.
            # Let's assume two distinct output tensors based on typical BlazeFace structure.
            if len(self.output_details) == 2:
                # Heuristic: scores tensor usually has fewer channels (1 or num_classes) than box tensor (4 coordinates)
                # And often scores are float32, boxes might also be float32
                # This is a guess; model inspection is key.
                if self.output_details[0]['shape'][-1] < self.output_details[1]['shape'][-1] : # e.g. [1,N,1] vs [1,N,4]
                    scores_tensor_index = self.output_details[0]['index']
                    boxes_tensor_index = self.output_details[1]['index']
                else:
                    scores_tensor_index = self.output_details[1]['index']
                    boxes_tensor_index = self.output_details[0]['index']
            elif len(self.output_details) == 1: # Could be a model with combined output or post-processing included
                logger.warning("Model has single output tensor. Decoding might be more complex or different.")
                # This case requires specific knowledge of the model's output structure
                # For now, we'll assume it's not the BlazeFace structure we're expecting for separate score/box.
                # Let's try to get it if it matches a common pattern, e.g. SSD-like output
                # output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                # raw_scores = output_data[..., NUM_CLASSES:] # Example slice
                # raw_boxes = output_data[..., :4] # Example slice
                # This is too model-specific to generalize here without more info.
                # For now, we'll just log and return empty if we don't find two expected tensors.
                logger.error("Unsupported output tensor configuration for this BlazeFace example.")
                return []
            else:
                logger.error(f"Unexpected number of output tensors: {len(self.output_details)}")
                return []


            raw_scores = self.interpreter.get_tensor(scores_tensor_index)
            raw_boxes = self.interpreter.get_tensor(boxes_tensor_index)

        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return []

        # Decode predictions (this part is highly model-specific, especially for BlazeFace)
        # The _decode_predictions method needs to be correctly implemented for BlazeFace
        # using its specific anchor logic. The current one is a placeholder.
        detections = self._decode_predictions(raw_boxes, raw_scores, original_shape)

        if len(detections) == 0:
            return []

        # Apply Non-Maximum Suppression
        final_detections = self._non_maximum_suppression(detections, self.iou_threshold)

        return final_detections.tolist() if len(final_detections) > 0 else []

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create a dummy image for testing (e.g., 640x480 BGR)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # You could draw a square to simulate a face if you want to test geometry
    # cv2.rectangle(dummy_image, (100, 100), (200, 200), (0, 255, 0), -1)


    # IMPORTANT: This test will likely FAIL or produce incorrect results
    # if the placeholder blazeface_frontend.tflite and anchors.npy are not replaced
    # with actual model and anchor files, and if the _decode_predictions method
    # is not correctly implemented for BlazeFace.
    logger.warning("Using placeholder model and anchor files. Detection results will not be meaningful.")

    # Create dummy model/anchor files if they don't exist, so it can run without erroring on file not found
    if not os.path.exists('src/models/blazeface_frontend.tflite'):
        with open('src/models/blazeface_frontend.tflite', 'w') as f: f.write("dummy tflite")
    if not os.path.exists('src/models/anchors.npy'):
        # Create a plausible anchor shape for testing structure, not for actual detection
        # BlazeFace typically has 896 anchors, each with 4 values (y, x, h, w)
        dummy_anchors = np.random.rand(896, 4).astype(np.float32)
        np.save('src/models/anchors.npy', dummy_anchors)
        logger.info("Created dummy anchors.npy for structural testing.")


    try:
        detector = FaceDetector(
            model_path='src/models/blazeface_frontend.tflite',
            anchors_path='src/models/anchors.npy'
        )

        # Test with a dummy image
        # Since the model is a placeholder, we can't expect real detections.
        # The goal here is to check if the class loads and runs without crashing.
        logger.info("Attempting to detect faces with placeholder model...")
        detections = detector.detect_faces(dummy_image)

        if detections:
            logger.info(f"Detected {len(detections)} 'faces' (using placeholder model):")
            for i, (xmin, ymin, xmax, ymax, score) in enumerate(detections):
                logger.info(f"  Face {i+1}: Box=({xmin},{ymin},{xmax},{ymax}), Score={score:.2f}")
                # Draw on dummy image (optional)
                # cv2.rectangle(dummy_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        else:
            logger.info("No 'faces' detected (as expected with placeholder model or if actual model fails).")

        # cv2.imshow("Detections on Dummy", dummy_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except FileNotFoundError as e:
        logger.error(f"Test failed: {e}. Ensure model and anchor files exist or are correctly pathed.")
    except RuntimeError as e:
        logger.error(f"Test failed during model loading or inference: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during testing: {e}")
    finally:
        # Clean up dummy files if they were created by this test script
        # This part is tricky because the main script also creates them.
        # For now, let's assume they are managed or replaced by actual files later.
        pass
