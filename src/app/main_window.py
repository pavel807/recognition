import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QInputDialog, QMessageBox, QSlider, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject

# Assuming these modules are structured correctly and accessible
# Need to handle potential import errors if files/classes are not ready
try:
    from src.core.camera import Camera
    from src.core.detection import FaceDetector
    # Placeholder paths for models, these should be configurable or exist
    # For UI development, we might not need fully functional models initially
    # but the classes should be instantiable.
    # Create dummy model/anchor files if they don't exist for detector
    import os
    models_dir = 'src/models'
    blazeface_model_path = os.path.join(models_dir, 'blazeface_frontend.tflite')
    anchors_path = os.path.join(models_dir, 'anchors.npy')
    mobilefacenet_model_path = os.path.join(models_dir, 'mobilefacenet.tflite')

    # Ensure dummy files exist for instantiation if real ones are missing
    os.makedirs(models_dir, exist_ok=True)
    if not os.path.exists(blazeface_model_path):
        with open(blazeface_model_path, 'w') as f: f.write("dummy")
    if not os.path.exists(anchors_path):
        # Basic anchors shape for BlazeFace (896 anchors, 4 coords)
        dummy_anchors = np.random.rand(896, 4).astype(np.float32)
        np.save(anchors_path, dummy_anchors)
    if not os.path.exists(mobilefacenet_model_path):
        with open(mobilefacenet_model_path, 'w') as f: f.write("dummy")

    from src.core.embedding import FeatureExtractor
    from src.core.recognition import RecognitionSystem
except ImportError as e:
    print(f"Error importing core modules: {e}. Ensure they are in PYTHONPATH or relative paths are correct.")
    # Fallback for UI development if core modules are problematic
    Camera = None
    FaceDetector = None
    FeatureExtractor = None
    RecognitionSystem = None
    # Show a message to the user that core components are missing.

DEFAULT_CAMERA_INDEX = 0
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# Worker thread for long-running CV tasks to avoid freezing the GUI
class RecognitionWorker(QObject):
    finished = pyqtSignal()
    results_ready = pyqtSignal(np.ndarray, list) # frame, recognition_results
    error_occurred = pyqtSignal(str)

    def __init__(self, camera, recognition_system):
        super().__init__()
        self.camera = camera
        self.recognition_system = recognition_system
        self._is_running = False

    def run(self):
        self._is_running = True
        while self._is_running:
            if not self.camera or not self.camera.is_opened():
                self.error_occurred.emit("Camera not available or not open.")
                QThread.msleep(100) # Wait a bit before retrying or stopping
                continue

            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                # self.error_occurred.emit("Failed to read frame from camera.")
                # This can happen often if camera is slow, don't spam errors
                QThread.msleep(30) # Approx 30 FPS
                continue

            try:
                # Process frame for recognition
                # Make a copy for drawing to avoid race conditions if frame is passed by reference elsewhere
                processed_frame = frame.copy()
                if self.recognition_system:
                    recognition_results = self.recognition_system.recognize_faces(processed_frame)
                    self.results_ready.emit(processed_frame, recognition_results)
                else:
                    # If no recognition system, just emit the frame
                    self.results_ready.emit(processed_frame, [])
            except Exception as e:
                self.error_occurred.emit(f"Error in recognition thread: {str(e)}")
                # Continue running unless it's a fatal error

            # QThread.msleep(10) # Adjust for desired frame rate, but processing time will dominate

    def stop(self):
        self._is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Facial Recognition")
        self.setGeometry(100, 100, 1000, 700) # x, y, width, height

        # Core components
        self.camera = None
        self.face_detector = None
        self.feature_extractor = None
        self.recognition_system = None
        self.recognition_worker = None
        self.recognition_thread = None

        self.is_camera_running = False

        self._init_core_components()
        self._init_ui()

        if not all([Camera, FaceDetector, FeatureExtractor, RecognitionSystem]):
            QMessageBox.critical(self, "Core Module Error",
                                 "Failed to load one or more core computer vision modules. "
                                 "Application functionality will be severely limited. "
                                 "Please check console for import errors.")


    def _init_core_components(self):
        try:
            if Camera:
                self.camera = Camera(DEFAULT_CAMERA_INDEX)
            # These paths should ideally be configurable
            if FaceDetector:
                self.face_detector = FaceDetector(
                    model_path=blazeface_model_path,
                    anchors_path=anchors_path
                )
            if FeatureExtractor:
                self.feature_extractor = FeatureExtractor(
                    model_path=mobilefacenet_model_path
                )
            if RecognitionSystem and self.face_detector and self.feature_extractor:
                self.recognition_system = RecognitionSystem(
                    self.face_detector,
                    self.feature_extractor,
                    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
                )
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error",
                                 f"Error initializing CV components: {e}\n"
                                 "Ensure models are correctly placed and valid.")
            # self.camera = None # Prevent further use if components failed
            # self.recognition_system = None


    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Video display
        self.video_label = QLabel("Camera feed will appear here.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        self.video_label.setMinimumSize(640, 480)
        self.layout.addWidget(self.video_label)

        # Controls
        self.controls_layout = QHBoxLayout()

        self.btn_toggle_camera = QPushButton("Start Camera")
        self.btn_toggle_camera.clicked.connect(self.toggle_camera)
        self.controls_layout.addWidget(self.btn_toggle_camera)

        self.btn_register_face = QPushButton("Register Face")
        self.btn_register_face.clicked.connect(self.register_face_dialog)
        self.btn_register_face.setEnabled(False) # Enable when camera is running
        self.controls_layout.addWidget(self.btn_register_face)

        self.layout.addLayout(self.controls_layout)

        # Similarity Threshold Slider
        form_layout = QFormLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100) # Representing 0.0 to 1.0
        self.threshold_slider.setValue(int(DEFAULT_SIMILARITY_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self.update_similarity_threshold)

        self.threshold_label = QLabel(f"Similarity Threshold: {DEFAULT_SIMILARITY_THRESHOLD:.2f}")
        form_layout.addRow(self.threshold_label, self.threshold_slider)
        self.layout.addLayout(form_layout)

        # Status Bar (using QLabel for simplicity here)
        self.status_label = QLabel("Status: Idle")
        self.layout.addWidget(self.status_label)

    def toggle_camera(self):
        if not self.camera:
            QMessageBox.warning(self, "Error", "Camera module not initialized.")
            return

        if not self.is_camera_running:
            if self.camera.open_camera(width=640, height=480): # Request a common resolution
                self.is_camera_running = True
                self.btn_toggle_camera.setText("Stop Camera")
                self.btn_register_face.setEnabled(True)
                self.status_label.setText("Status: Camera Running")

                # Start worker thread for recognition
                if self.recognition_system and self.camera:
                    self.recognition_thread = QThread()
                    self.recognition_worker = RecognitionWorker(self.camera, self.recognition_system)
                    self.recognition_worker.moveToThread(self.recognition_thread)

                    self.recognition_worker.results_ready.connect(self.update_frame)
                    self.recognition_worker.finished.connect(self.recognition_thread.quit)
                    self.recognition_worker.finished.connect(self.recognition_worker.deleteLater)
                    self.recognition_thread.finished.connect(self.recognition_thread.deleteLater)
                    self.recognition_worker.error_occurred.connect(self.handle_worker_error)

                    self.recognition_thread.started.connect(self.recognition_worker.run)
                    self.recognition_thread.start()
                else: # Fallback if no recognition system, just show camera feed
                    self.fallback_timer = QTimer(self)
                    self.fallback_timer.timeout.connect(self.update_frame_fallback)
                    self.fallback_timer.start(30) # ~33 FPS

            else:
                QMessageBox.warning(self, "Camera Error", "Failed to open camera.")
                self.status_label.setText("Status: Error - Camera failed to open")
        else:
            self.is_camera_running = False
            if self.recognition_worker:
                self.recognition_worker.stop()
            if self.recognition_thread and self.recognition_thread.isRunning():
                self.recognition_thread.quit()
                self.recognition_thread.wait() # Wait for thread to finish

            if hasattr(self, 'fallback_timer') and self.fallback_timer.isActive():
                self.fallback_timer.stop()

            if self.camera:
                self.camera.close_camera()

            self.btn_toggle_camera.setText("Start Camera")
            self.btn_register_face.setEnabled(False)
            self.video_label.setText("Camera stopped.") # Clear video feed
            self.video_label.setPixmap(QPixmap()) # Clear image
            self.status_label.setText("Status: Camera Stopped")

    def handle_worker_error(self, error_message):
        # Avoid flooding with messages, maybe show in status bar or log
        self.status_label.setText(f"Status: Worker Error - {error_message}")
        print(f"RecognitionWorker Error: {error_message}") # Log to console
        # Optionally, stop the camera or worker if error is critical
        # self.toggle_camera() # This would stop the camera

    def update_frame_fallback(self):
        """Fallback for displaying camera feed if recognition system is not available."""
        if not self.camera or not self.camera.is_opened():
            return
        ret, frame = self.camera.read_frame()
        if ret and frame is not None:
            self.display_image(frame)

    def update_frame(self, frame, recognition_results):
        if frame is None:
            return

        processed_frame = frame.copy() # Ensure we work on a copy

        # Overlay recognition results
        for result in recognition_results:
            xmin, ymin, xmax, ymax = result['bbox']
            name = result['name']
            similarity = result.get('similarity', 0.0) # Use .get for safety
            detection_score = result.get('score', 0.0)

            color = QColor(0, 255, 0) # Green for known
            if name == "Unknown":
                color = QColor(255, 0, 0) # Red for unknown
            elif name == "ErrorInCrop" or name == "EmptyCrop":
                color = QColor(255,165,0) # Orange for crop errors

            # Draw bounding box
            painter = QPainter() # Will be used on QPixmap
            # This drawing logic needs to be adapted to draw on QPixmap before setting it to QLabel
            # For now, let's use cv2 for drawing on the numpy array directly
            cv2.rectangle(processed_frame, (xmin, ymin), (xmax, ymax), color.getRgb()[:3], 2)

            text = f"{name}"
            if name != "Unknown" and name != "ErrorInCrop" and name != "EmptyCrop":
                 text += f" ({similarity:.2f})"

            # Put text above the box
            cv2.putText(processed_frame, text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.getRgb()[:3], 2)

        self.display_image(processed_frame)


    def display_image(self, cv_img):
        """Converts an OpenCV image to QPixmap and displays it."""
        if cv_img is None:
            self.video_label.clear()
            return

        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.video_label.setText("Error displaying frame.")


    def register_face_dialog(self):
        if not self.is_camera_running or not self.camera or not self.recognition_system:
            QMessageBox.warning(self, "Registration Error", "Camera must be running to register a face.")
            return

        ret, frame = self.camera.read_frame() # Get current frame
        if not ret or frame is None:
            QMessageBox.warning(self, "Registration Error", "Could not capture frame for registration.")
            return

        name, ok = QInputDialog.getText(self, "Register Face", "Enter name for the face:")
        if ok and name:
            # Potentially run this in a separate thread if it's slow, but registration is usually a one-off
            success = self.recognition_system.register_face(frame, name)
            if success:
                QMessageBox.information(self, "Registration Success", f"Face for '{name}' registered.")
                self.status_label.setText(f"Status: Registered '{name}'")
            else:
                QMessageBox.warning(self, "Registration Failed", f"Could not register face for '{name}'. Check logs/console.")
                self.status_label.setText(f"Status: Failed to register '{name}'")
        elif ok and not name:
             QMessageBox.warning(self, "Input Error", "Name cannot be empty.")


    def update_similarity_threshold(self, value):
        threshold = value / 100.0
        self.threshold_label.setText(f"Similarity Threshold: {threshold:.2f}")
        if self.recognition_system:
            self.recognition_system.set_similarity_threshold(threshold)
        self.status_label.setText(f"Status: Threshold set to {threshold:.2f}")

    def closeEvent(self, event):
        """Ensure camera and threads are stopped when closing the window."""
        if self.is_camera_running:
            self.toggle_camera() # This should handle stopping worker and camera

        # Wait for thread to surely finish if it was running
        if self.recognition_thread and self.recognition_thread.isRunning():
            self.recognition_thread.quit()
            if not self.recognition_thread.wait(1000): # Wait up to 1 sec
                print("Recognition thread did not stop gracefully.")

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
