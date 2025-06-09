from flask import Flask
import os

# Create a global instance of the app
app = Flask(__name__)

# Configuration (optional for now, can be expanded)
# Example: app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '..', 'uploads')
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import routes after app initialization to avoid circular imports
from . import routes

# Initialize database (optional here, can be done on first request or app start)
# from .database_utils import init_db
# init_db()
# print("Database initialized from __init__.py")
