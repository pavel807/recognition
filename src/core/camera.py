import cv2
import logging

logger = logging.getLogger(__name__)

class Camera:
    """
    Handles webcam capture using OpenCV.
    """
    def __init__(self, camera_index: int = 0):
        """
        Initializes the camera.

        Args:
            camera_index (int): The index of the camera to use.
        """
        self.camera_index = camera_index
        self.cap = None
        self.default_width = 640
        self.default_height = 480
        self.default_fps = 30

    def open_camera(self, width: int = None, height: int = None, fps: int = None) -> bool:
        """
        Opens the camera and sets desired properties.

        Args:
            width (int, optional): Desired frame width. Defaults to self.default_width.
            height (int, optional): Desired frame height. Defaults to self.default_height.
            fps (int, optional): Desired frames per second. Defaults to self.default_fps.

        Returns:
            bool: True if the camera was opened successfully, False otherwise.
        """
        if self.is_opened():
            logger.warning("Camera is already open. Closing existing capture before reopening.")
            self.close_camera()

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera with index {self.camera_index}")
            self.cap = None
            return False

        # Set desired properties
        target_width = width if width is not None else self.default_width
        target_height = height if height is not None else self.default_height
        target_fps = fps if fps is not None else self.default_fps

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        # Verify settings (OpenCV might not always set them exactly)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera opened. Requested: {target_width}x{target_height} @ {target_fps}FPS. "
                    f"Actual: {actual_width}x{actual_height} @ {actual_fps:.2f}FPS.")

        # If actual dimensions are zero, it's likely an issue
        if actual_width == 0 or actual_height == 0:
            logger.error(f"Camera {self.camera_index} returned 0x0 frame size. Might be an issue with the camera or drivers.")
            self.close_camera()
            return False

        return True

    def is_opened(self) -> bool:
        """
        Checks if the camera is currently open.

        Returns:
            bool: True if the camera is open, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        """
        Reads a frame from the camera.

        Returns:
            tuple: (bool, numpy.ndarray) where bool indicates success
                   and numpy.ndarray is the frame (BGR format).
                   Returns (False, None) if the camera is not open or frame read fails.
        """
        if not self.is_opened():
            logger.error("Camera not open. Cannot read frame.")
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            return False, None
        return ret, frame

    def close_camera(self):
        """
        Releases the camera capture.
        """
        if self.cap is not None:
            logger.info("Closing camera.")
            self.cap.release()
            self.cap = None

    def get_properties(self) -> dict:
        """
        Gets current camera properties.

        Returns:
            dict: A dictionary containing frame width, frame height, and FPS.
                  Returns empty dict if camera is not open.
        """
        if not self.is_opened():
            return {}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS)
        }

if __name__ == '__main__':
    # Example Usage (for testing)
    logging.basicConfig(level=logging.INFO)

    camera = Camera(camera_index=0)
    if camera.open_camera(width=1280, height=720, fps=30):
        logger.info(f"Camera properties: {camera.get_properties()}")

        for _ in range(10): # Capture a few frames
            ret, frame = camera.read_frame()
            if ret:
                logger.info(f"Frame captured, shape: {frame.shape}")
                # In a real app, you would display or process the frame here
                # cv2.imshow("Frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                logger.error("Failed to capture frame.")
                break

        camera.close_camera()
        # cv2.destroyAllWindows()
    else:
        logger.error("Could not open camera.")
