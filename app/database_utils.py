import sqlite3
import numpy as np
import io
import os # For path manipulation

# Path to the database file, one level up from 'app' directory, then into 'database'
DB_FILE = os.path.join(os.path.dirname(__file__), '..', 'database', 'faces.db')
DATABASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'database')

# Adapters for numpy arrays
def adapt_array(arr):
    # Convert numpy array to TEXT
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read()) # Store as BLOB

def convert_array(text):
    # Convert BLOB from SQLite back to numpy array
    out = io.BytesIO(text)
    out.seek(0)
    # Allow pickle since we are controlling the data source
    return np.load(out, allow_pickle=True)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array) # Use "ARRAY" as the type name

def init_db():
    # Ensure the database directory exists
    os.makedirs(DATABASE_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding ARRAY NOT NULL, -- Custom type for numpy array
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized/checked at {DB_FILE}")

def add_face(name, encoding, image_path=None):
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO known_faces (name, encoding, image_path)
            VALUES (?, ?, ?)
        ''', (name, encoding, image_path))
        conn.commit()
        print(f"Added face for {name}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def get_all_known_faces():
    # Ensure DB exists before trying to fetch
    if not os.path.exists(DB_FILE):
        print("Database file not found. Initializing.")
        init_db() # Create db and table if they don't exist

    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    known_faces = []
    try:
        cursor.execute("SELECT name, encoding FROM known_faces")
        rows = cursor.fetchall()
        for row in rows:
            name = row[0]
            encoding = row[1] # This should be automatically converted by convert_array
            known_faces.append({'name': name, 'encoding': encoding})
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        # If the table doesn't exist (e.g. "no such table: known_faces"), initialize and return empty
        if "no such table" in str(e).lower():
            print("Table 'known_faces' not found. Initializing database.")
            init_db() # Initialize and try again (or just return empty for this call)
            return [] # Return empty list after attempting init_db on error

    finally:
        conn.close()
    return known_faces

if __name__ == '__main__':
    # Example Usage (for testing the script directly)
    init_db() # Ensure database and table are created

    # Create a dummy encoding (replace with actual face encoding)
    dummy_encoding = np.random.rand(128)
    dummy_encoding2 = np.random.rand(128)

    add_face("Test User 1", dummy_encoding, "path/to/image1.jpg")
    add_face("Test User 2", dummy_encoding2)

    faces = get_all_known_faces()
    if faces:
        for face in faces:
            print(f"Retrieved: Name: {face['name']}, Encoding shape: {face['encoding'].shape}")
            # print(f"Retrieved: Name: {face[0]}, Encoding: {face[1]}")
    else:
        print("No faces found in the database.")

    # Test with a non-existent DB to ensure it gets created
    # if os.path.exists(DB_FILE):
    #     os.remove(DB_FILE)
    #     print("Removed DB for testing, run again.")
    # else:
    #     init_db()
    #     add_face("Test User After Delete", np.random.rand(128))
    #     faces = get_all_known_faces()
    #     for face in faces:
    #         print(f"Retrieved after re-init: Name: {face['name']}")
