import cv2
import face_recognition
import numpy as np
from .face_utils import get_face_encodings, find_best_match # Use existing utils
from .database_utils import get_all_known_faces # To get known faces

class VideoCamera:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise IOError("Cannot open webcam")

        # Load known faces from the database ONCE at initialization for efficiency
        # This means new faces uploaded while the camera is running won't be recognized
        # until the camera class is re-initialized (e.g., app restart or a more complex refresh mechanism).
        self.known_faces_data = get_all_known_faces()
        print(f"VideoCamera initialized with {len(self.known_faces_data)} known faces.")

    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        print("VideoCamera released webcam.")

    def get_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return None # Or raise an exception / return a placeholder image

        # Resize frame for faster processing (optional)
        # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        current_face_locations = face_recognition.face_locations(rgb_frame)
        current_face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=current_face_locations)

        face_names = []
        for face_encoding in current_face_encodings:
            # See if the face is a match for the known face(s)
            name = find_best_match(self.known_faces_data, face_encoding)
            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(current_face_locations, face_names):
            # Scale back up face locations if frame was resized
            # top *= 2; right *= 2; bottom *= 2; left *= 2 # If using small_frame

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) # Red

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return None # Or handle error

        return jpeg.tobytes()

    def refresh_known_faces(self):
        # Method to manually refresh known faces from DB if needed later
        self.known_faces_data = get_all_known_faces()
        print(f"VideoCamera known faces refreshed: {len(self.known_faces_data)} faces.")

# Remove or comment out the old main() function if it exists
# if __name__ == '__main__':
#    # This part is for standalone testing of camera.py, not for Flask app
#    cam = VideoCamera()
#    print("Standalone test: Starting camera. Press 'q' in OpenCV window to quit.")
#    while True:
#        frame_bytes = cam.get_frame() # This returns JPEG bytes
#        if frame_bytes:
#            # To display it in a window, you'd decode it
#            frame_display = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
#            cv2.imshow('Standalone Test - Press q to quit', frame_display)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#    cv2.destroyAllWindows()
#    del cam # Explicitly delete to trigger __del__ for resource release
