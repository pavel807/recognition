# Real-Time Facial Recognition System for macOS

This project is a real-Time facial recognition system developed in Python, using Qt (PyQt5) for the graphical user interface, TensorFlow Lite for model inference, and OpenCV for camera interactions. It's designed with macOS optimizations in mind, including support for TensorFlow-Metal and the Core ML delegate.

## Features

-   **Real-Time Video Processing:** Captures video from webcam.
-   **Face Detection:** Utilizes a TFLite model (e.g., BlazeFace) to detect faces in the video stream.
-   **Face Embedding:** Employs a TFLite model (e.g., MobileFaceNet) to generate feature embeddings for detected faces.
-   **Face Registration:** Allows users to register faces with names, storing their embeddings.
-   **Face Recognition:** Compares detected faces against the registered database to identify individuals.
-   **Responsive GUI:** Built with PyQt5, featuring:
    -   Live camera feed display.
    -   Overlay of bounding boxes and recognized names.
    -   Controls for starting/stopping the camera and registering new faces.
    -   Adjustable similarity threshold for recognition.
-   **Performance Optimizations for macOS:**
    -   GPU acceleration via `tensorflow-metal` (for Apple Silicon / AMD GPUs).
    -   Support for Core ML delegate for TFLite models for potential Neural Engine/GPU acceleration.
-   **Cross-Platform Core:** While UI and some optimizations target macOS, the core CV pipeline is Python-based.
-   **Packaging Support:** Includes `setup.py` for `py2app` to create a standalone macOS application bundle.

## Project Structure

```
.
├── data/                  # (Optional) For persistent registered faces database (not implemented yet)
├── src/
│   ├── app/               # Qt application logic
│   │   └── main_window.py
│   ├── core/              # Core computer vision pipeline
│   │   ├── camera.py
│   │   ├── detection.py
│   │   ├── embedding.py
│   │   └── recognition.py
│   ├── models/            # Placeholder for TFLite models (blazeface_frontend.tflite, anchors.npy, mobilefacenet.tflite)
│   │                       # You need to replace these with actual model files.
│   ├── utils/             # Helper functions
│   │   ├── __init__.py
│   │   └── resource_path.py
│   └── __init__.py        # Makes src a package (optional)
├── tests/
│   └── test_core_components.py # Unit tests
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── setup.py               # For py2app packaging
```

## Prerequisites

-   macOS (tested on macOS Catalina 10.15 and newer, but should work on compatible versions).
-   Python 3.8+
-   A webcam.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # .\venv\Scripts\activate   # On Windows (though this project is macOS focused)
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install OpenCV, TensorFlow, PyQt5, `tensorflow-metal` (for macOS GPU), and other necessary packages.

4.  **Obtain TFLite Models:**
    This project uses placeholder files in `src/models/`. You **must** replace them with actual TFLite models for face detection and feature extraction.
    -   **Face Detection (e.g., BlazeFace):**
        -   `src/models/blazeface_frontend.tflite`
        -   `src/models/anchors.npy` (if required by your BlazeFace model for post-processing)
    -   **Feature Extraction (e.g., MobileFaceNet):**
        -   `src/models/mobilefacenet.tflite`

    You can find pre-trained models from sources like TensorFlow Hub, MediaPipe, or other model zoos. Ensure the model input/output specifications match what's expected by `src/core/detection.py` and `src/core/embedding.py` or update the preprocessing/postprocessing code accordingly. The current BlazeFace implementation in `detection.py` has a **placeholder decoding logic** and needs to be correctly implemented for the specific model you choose.

## Running the Application (Development)

Once dependencies are installed and models are in place:

```bash
python src/app/main_window.py
```

This will launch the Qt application.

## Usage

1.  **Start Camera:** Click the "Start Camera" button. The video feed should appear.
2.  **Register Face:**
    -   Position a face clearly in the camera view.
    -   Click "Register Face".
    -   A dialog will prompt you to enter a name for the person.
    -   Click "OK". A message will confirm if registration was successful.
3.  **Recognition:** Once faces are registered, the system will attempt to recognize them in the live video feed.
    -   Detected faces will have a bounding box.
    -   Recognized individuals will be labeled with their name and a similarity score.
    -   Unknown faces will be labeled "Unknown".
4.  **Similarity Threshold:** Adjust the slider to change the sensitivity of recognition. A lower threshold makes recognition easier (more potential false positives), while a higher threshold requires a closer match.
5.  **Stop Camera:** Click "Stop Camera" to end the video stream.

## Packaging for macOS (Standalone Application)

The project is configured to be packaged into a standalone `.app` bundle using `py2app`.

1.  **Ensure `py2app` is installed** (it's in `requirements.txt`):
    ```bash
    pip install py2app
    ```
2.  **Build the application:**
    From the project root directory, run:
    ```bash
    python setup.py py2app
    ```
    This will create a `dist` folder containing `FacialRecognitionApp.app`.

3.  **Run the bundled application:**
    Double-click `FacialRecognitionApp.app` in the `dist` folder.

    **Note on Packaging:** Packaging complex applications with dependencies like TensorFlow can sometimes be tricky. If you encounter issues, you might need to adjust the `OPTIONS` in `setup.py` (e.g., `includes`, `packages`, `resources`). Test the bundled app thoroughly.

## Running Unit Tests

To run the basic unit tests:

```bash
python -m unittest tests.test_core_components
```
Or, if `tests` is treated as a discoverable package:
```bash
python -m unittest discover tests
```

## TODO / Potential Future Enhancements

-   Implement robust decoding logic for the chosen BlazeFace model in `src/core/detection.py`.
-   Replace placeholder TFLite models with actual, high-performance models.
-   Persistent storage for registered faces (e.g., SQLite, JSON file).
-   More advanced UI features (e.g., gallery of registered faces, configuration panel).
-   Performance profiling and more in-depth optimization.
-   Comprehensive error logging to a file.
-   Support for multiple cameras.
-   Batch inference for feature extraction if multiple faces are detected.
-   More sophisticated face tracking between frames.

## Troubleshooting

-   **Camera Not Opening:**
    -   Ensure your webcam is connected and functional.
    -   Check if another application is using the camera.
    -   On macOS, ensure the application (or your terminal if running from source) has camera permissions (System Settings > Privacy & Security > Camera).
-   **"Model file not found" errors:** Ensure you have placed the correct `.tflite` and `anchors.npy` files in `src/models/`. Ensure that `src/utils/resource_path.py` is correctly resolving paths for your environment.
-   **Low Performance:**
    -   If on Apple Silicon, ensure `tensorflow-metal` is installed and working (check console logs at app startup).
    -   The Core ML delegate might offer better performance for TFLite models; this is attempted automatically on macOS (check console logs).
    -   The chosen models might be too heavy for your hardware. Consider lighter alternatives.
    -   The placeholder models/decoding logic currently in the project will not perform well and are not functional for real recognition.
-   **Packaging Issues:** Refer to `py2app` documentation and try adjusting `setup.py` options. Pay close attention to `packages`, `includes`, and how resources like models are handled. Test on a clean machine if possible.

---
This README provides a good starting point for users and developers.
