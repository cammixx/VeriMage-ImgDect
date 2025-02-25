from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import logging
from PIL import Image
# from detector import DummyDetector  # Import the dummy detector

# Initialize Flask application
app = Flask(__name__)

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application Configuration
app.config.update(
    # Upload settings
    UPLOAD_FOLDER='static/uploads',  # Where uploaded images are stored
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'},  # Allowed file types
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # Maximum file size (16MB)
    
    # Image quality requirements
    MIN_IMAGE_SIZE=(224, 224),  # Minimum dimensions required (width, height)
    MIN_RESOLUTION_DPI=72,  # Minimum image resolution
    
    # AI Model settings
    MODEL_CACHE_DIR='models'  # Where to cache AI models
)

# Initialize the image detector
# TODO: Replace DummyDetector with your actual detector implementation
#detector = DummyDetector()

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image_quality(file_stream) -> tuple[bool, str]:
    """Validate image dimensions and resolution.
    
    Args:
        file_stream: File-like object containing the image
        
    Returns:
        tuple: (is_valid, message)
            - is_valid (bool): True if image meets quality requirements
            - message (str): Description of validation result or error
    """
    try:
        # Open image and get its properties
        image = Image.open(file_stream)
        width, height = image.size
        dpi = image.info.get('dpi', (0, 0))[0]
        
        # Check minimum dimensions
        if width < app.config['MIN_IMAGE_SIZE'][0] or height < app.config['MIN_IMAGE_SIZE'][1]:
            return False, f"Image too small. Minimum size is {app.config['MIN_IMAGE_SIZE'][0]}x{app.config['MIN_IMAGE_SIZE'][1]} pixels"
        
        # Check resolution if DPI info is available
        if dpi and dpi < app.config['MIN_RESOLUTION_DPI']:
            return False, f"Image resolution too low. Minimum is {app.config['MIN_RESOLUTION_DPI']} DPI"
        
        file_stream.seek(0)  # Reset file pointer for subsequent operations
        return True, "Image meets quality requirements"
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False, "Error validating image quality"

# API Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload, validation, and detection.
    
    Expects a file in the request with key 'file'.
    
    Returns:
        JSON response with upload result and detection results if successful.
        Error message with appropriate status code if upload fails.
    """
    try:
        # Check if file was included in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
            
        if file and allowed_file(file.filename):
            # Validate image quality
            is_valid, message = validate_image_quality(file)
            if not is_valid:
                return jsonify({"error": message}), 400
            
            # Save the file
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Run detection on the uploaded image
            detection_results = detector.detect(filepath)
            
            logger.info(f"File processed successfully: {filename}")
            return jsonify({
                "message": "File processed successfully",
                "filename": filename,
                "detection_results": detection_results
            }), 200
            
        return jsonify({"error": "Invalid file type"}), 400
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_CACHE_DIR'], exist_ok=True)
    
    # Start the Flask development server
    app.run(debug=True)