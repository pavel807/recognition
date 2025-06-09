from . import app # Imports the app instance from app/__init__.py

if __name__ == '__main__':
    # When using a global camera object and the Flask reloader (debug=True),
    # the camera might be initialized multiple times or not released properly.
    # Setting use_reloader=False can help if you encounter issues like:
    # - "Webcam already in use"
    # - The camera feed freezing after a code change.
    # However, you lose the auto-reloading convenience during development.
    # For robust webcam handling, especially in production, other strategies are needed.
    app.run(debug=True, host='0.0.0.0', port=5000) # Default: use_reloader=True
    # app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) # Alternative if issues
