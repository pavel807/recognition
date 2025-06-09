import face_recognition
import numpy as np
# from .database_utils import get_all_known_faces # Avoid direct db import here, pass data instead

def get_face_encodings(rgb_image):
    """
    Takes an RGB image, detects faces, and returns their encodings.
    :param rgb_image: Numpy array of an image in RGB format.
    :return: A list of 128-dimension face encodings (numpy arrays).
    """
    # Detect face locations using the default HOG model.
    # For more accuracy, model='cnn' can be used, but it's more resource-intensive.
    face_locations = face_recognition.face_locations(rgb_image)

    # Compute face encodings for the detected locations.
    # This returns a list of 128-dimensional face encodings.
    encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)

    return encodings

def find_best_match(known_faces_data, unknown_encoding, tolerance=0.6):
    """
    Compares an unknown face encoding against a list of known faces and finds the best match.

    :param known_faces_data: A list of dictionaries, where each dictionary has
                             'name' (str) and 'encoding' (numpy.ndarray).
                             Example: [{'name': 'Person A', 'encoding': array([...])}, ...]
    :param unknown_encoding: A single face encoding (numpy.ndarray) of the face to be identified.
    :param tolerance: How much distance between faces to consider it a match. Lower is stricter.
                      Default is 0.6.
    :return: The name of the best matching known face, or "Unknown" if no match is found.
    """
    if not known_faces_data:
        return "Unknown"

    known_encodings = [data['encoding'] for data in known_faces_data]
    known_names = [data['name'] for data in known_faces_data]

    # Compare the unknown encoding with all known encodings
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=tolerance)

    name = "Unknown"

    # If there's at least one match
    if True in matches:
        # Calculate face distances for all known_encodings to the unknown_encoding
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

        # Find the index of the best match (minimum distance)
        # We only consider distances for faces that were True matches
        best_match_index = -1
        min_distance = float('inf')

        for i, (match, distance) in enumerate(zip(matches, face_distances)):
            if match and distance < min_distance:
                min_distance = distance
                best_match_index = i

        if best_match_index != -1:
            name = known_names[best_match_index]
            # print(f"Match found: {name} with distance: {min_distance}") # For debugging

    return name

if __name__ == '__main__':
    # This block is for testing purposes.
    # You would need to populate known_faces_data with actual encodings and names.

    print("Face utils script. To test, run with a script that provides images and known faces data.")

    # Example of how it might be used (requires actual image data and populated known_faces_data):
    # 1. Load known faces from database (this would typically happen in your main app logic)
    #    known_faces_from_db = [
    #        {'name': 'Person 1', 'encoding': np.random.rand(128)}, # Replace with actual encoding
    #        {'name': 'Person 2', 'encoding': np.random.rand(128)}  # Replace with actual encoding
    #    ]

    # 2. Get an encoding from a new image (e.g., from webcam)
    #    # Create a dummy RGB image (3 channels)
    #    # In a real scenario, this would come from cv2.imread() and then cv2.cvtColor(..., cv2.COLOR_BGR2RGB)
    #    # or directly from a webcam frame that's been converted to RGB.
    #    dummy_rgb_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    #
    #    # Simulate that face_recognition.face_encodings returns a list, even if one face.
    #    # And that the encoding is a 128-dim array.
    #    # In reality, get_face_encodings would call face_recognition library.
    #    # For this test, we'll mock a plausible output from get_face_encodings.
    #    # Let's assume get_face_encodings found one face and returned its encoding:
    #    # For this self-contained test, we'll just create a dummy unknown encoding directly.
    #    # unknown_face_encodings_list = get_face_encodings(dummy_rgb_image) # This would call the actual library

    #    # Let's create a dummy unknown encoding that might match one of the known faces or not.
    #    unknown_test_encoding = np.random.rand(128) # A new random face
    #    # To test a match, you might make it very similar to an existing one:
    #    # unknown_test_encoding = known_faces_from_db[0]['encoding'] + np.random.uniform(-0.1, 0.1, 128)


    #    if known_faces_from_db: # Ensure there are known faces to compare against
    #       # Simulate having one unknown encoding to test
    #       # if unknown_face_encodings_list: # If get_face_encodings found any faces
    #       #    current_unknown_encoding = unknown_face_encodings_list[0]
    #           name_of_match = find_best_match(known_faces_from_db, unknown_test_encoding)
    #           print(f"The unknown face was identified as: {name_of_match}")
    #       # else:
    #       #    print("No faces found in the dummy unknown image.")
    #    else:
    #        print("No known faces to compare against.")
