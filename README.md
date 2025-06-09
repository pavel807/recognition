# Real-Time Face Recognition System with Web UI

A real-time face recognition system that captures live video from a webcam, detects faces, compares them against a database of known individuals, and displays the results through a Flask-based web interface.

## Features

*   **Live Webcam Feed:** Displays real-time video from the primary webcam.
*   **Face Detection:** Automatically detects faces in the video stream.
*   **Face Recognition:** Identifies detected faces by comparing them against a database of known faces.
*   **Database Storage:** Stores face encodings and names of known individuals using SQLite.
*   **Web Interface:**
    *   View live webcam feed with recognized names overlaid.
    *   Upload images of new individuals to add them to the recognition database.
    *   View a list of all individuals currently in the database, along with their uploaded images.
*   **Modern UI Theme:** A clean, dark-themed "futuristic" style interface.

## Directory Structure

```
├── app/
│   ├── static/              # (Currently unused, for static CSS/JS if needed)
│   ├── templates/           # HTML templates for Flask
│   │   ├── index.html       # Main page with webcam feed
│   │   ├── list_faces.html  # Page to display known faces
│   │   └── upload.html      # Page for uploading new faces
│   ├── __init__.py          # Initializes the Flask app
│   ├── camera.py            # VideoCamera class for webcam access & processing
│   ├── database_utils.py    # Utilities for SQLite database interaction
│   ├── face_utils.py        # Utilities for face encoding and comparison
│   ├── main.py              # Main Flask application runner (entry point)
│   └── routes.py            # Flask routes for web pages and API endpoints
├── database/
│   └── faces.db             # SQLite database file (created automatically)
├── uploads_for_db/          # Directory where uploaded images are stored (created automatically)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup Instructions

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   A C++ compiler, CMake, and Python development headers are required for installing `dlib`, a dependency of `face_recognition`.
    *   On Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y build-essential cmake python3-dev`
    *   Other systems may require different commands.

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    *Note: During development of this project, direct global installation of packages was sometimes used due to subtask environment limitations. A virtual environment is standard best practice.*

3.  **Install Dependencies:**
    Ensure you have installed the prerequisites for `dlib` mentioned above.
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    If you encounter issues with `dlib` installation, ensure `cmake` and `python3-dev` (or equivalent for your OS) are correctly installed and accessible.

4.  **Database Initialization:**
    The SQLite database (`database/faces.db`) and the table for known faces will be created automatically when the application starts for the first time. The `uploads_for_db` directory will also be created automatically.

## Running the Application

1.  **Start the Flask server:**
    Navigate to the project's root directory (where `requirements.txt` is located).
    If you are using a virtual environment, make sure it's activated.
    ```bash
    python -m app.main
    ```

2.  **Access the Web Interface:**
    Open your web browser and go to:
    [http://localhost:5000](http://localhost:5000)
    (Or `http://<your-server-ip>:5000` if running on a remote machine and firewall configured).

## Usage

*   **Main Page (`/`):** Displays the live webcam feed. If faces are detected and recognized (based on uploaded images), their names will be shown.
*   **Upload Known Faces (`/upload`):**
    1.  Navigate to the "Upload Known Faces" page.
    2.  Enter the name of the individual.
    3.  Choose an image file containing a clear view of their face.
    4.  Click "Upload Face".
    5.  The system will process the image, extract the face encoding, and store it in the database.
    6.  You may need to click the "Refresh Camera Faces" button on the main page for the live feed to immediately recognize newly uploaded faces.
*   **View Known Faces (`/list_faces`):** Shows a list of all individuals whose faces have been uploaded, along with their images.

## Key Technologies Used

*   **Python:** Core programming language.
*   **Flask:** Web framework for the user interface and API.
*   **OpenCV (`cv2`):** For webcam access and image processing.
*   **face_recognition:** For face detection and generating face encodings.
*   **dlib:** A dependency of `face_recognition` for the core machine learning algorithms.
*   **NumPy:** For numerical operations, especially with image and encoding data.
*   **SQLite:** For the database to store face information.
*   **HTML/CSS:** For structuring and styling the web interface.
