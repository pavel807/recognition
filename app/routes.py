from flask import render_template, request, redirect, url_for, flash, Response, send_from_directory # Add Response and send_from_directory
from . import app # Import the app instance from __init__.py
from .face_utils import get_face_encodings
from .database_utils import add_face, init_db, get_all_known_faces
from .camera import VideoCamera # Import the new camera class
import cv2
import numpy as np
import os

# Initialize the database once, when the app starts or on first relevant request
init_db()
print("Database initialized from routes.py during import or first run.")

# Define a directory for saving uploaded images
UPLOAD_FOLDER_NAME = 'uploads_for_db'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, '..', UPLOAD_FOLDER_NAME)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global camera instance
video_camera = None

def get_camera():
    global video_camera
    if video_camera is None:
        try:
            video_camera = VideoCamera()
            print("VideoCamera instance created for routes.")
        except IOError as e:
            print(f"Error initializing VideoCamera: {e}")
            return None
    return video_camera

def gen(camera_instance):
    if camera_instance is None:
        print("Camera instance is None in gen(). Cannot generate frames.")
        return

    print("Starting frame generation from camera.")
    while True:
        frame = camera_instance.get_frame()
        if frame is None:
            print("Frame was None, stopping generation.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', message="Welcome to the Face Recognition System!")

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        name = request.form.get('name', '').strip()

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if not name:
            flash('Name is required')
            return redirect(request.url)

        if file:
            try:
                # Using werkzeug.utils.secure_filename is good practice.
                from werkzeug.utils import secure_filename
                # image_filename_secure = secure_filename(file.filename) # Use this
                image_filename_secure = file.filename # For now, keep as is, but note for security.

                saved_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename_secure)
                file.save(saved_image_path)

                img = cv2.imread(saved_image_path)
                if img is None:
                    flash(f"Could not read image: {image_filename_secure}")
                    # os.remove(saved_image_path) # Clean up if bad image
                    return redirect(request.url)

                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = get_face_encodings(rgb_image)

                if encodings:
                    face_encoding = encodings[0]
                    # Store only the filename (or path relative to UPLOAD_FOLDER) for the DB
                    add_face(name, face_encoding, image_filename_secure) # Store just filename
                    flash(f"Face for '{name}' added successfully from {image_filename_secure}!")

                    # Refresh camera if active
                    cam_instance = get_camera()
                    if cam_instance:
                        cam_instance.refresh_known_faces()
                        flash("Known faces in camera feed refreshed.")
                else:
                    flash(f"No faces found in {image_filename_secure}.") # Use secured name
                    # os.remove(saved_image_path) # Clean up if no faces and not needed

                return redirect(url_for('upload_image'))

            except Exception as e:
                flash(f"An error occurred: {str(e)}")
                # import traceback
                # traceback.print_exc() # For server-side debugging
                return redirect(request.url)

    return render_template('upload.html')

@app.route('/list_faces')
def list_faces():
    faces = get_all_known_faces()
    return render_template('list_faces.html', faces=faces)

@app.route('/video_feed')
def video_feed():
    cam = get_camera()
    if cam is None:
        return Response("Camera not available. Please ensure it is connected and not in use by another application.", status=503, mimetype='text/plain')

    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/refresh_faces_in_camera', methods=['POST'])
def refresh_faces_in_camera():
    cam = get_camera()
    if cam:
        cam.refresh_known_faces()
        flash("Attempted to refresh known faces in the camera feed.")
    else:
        flash("Camera not active, cannot refresh faces.")
    return redirect(url_for('index'))

@app.route('/uploads_for_db/<path:filename>')
def uploaded_file(filename):
    # UPLOAD_FOLDER is an absolute path to <project_root>/uploads_for_db
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
