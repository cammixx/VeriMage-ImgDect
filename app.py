from flask import Flask, request, jsonify, render_template, send_from_directory, make_response, url_for, redirect
import os
from werkzeug.utils import secure_filename
import logging
from PIL import Image
import traceback  # Added for better error tracking
import datetime
import joblib
import numpy as np
from flask_cors import CORS
import base64
from sklearn.preprocessing import StandardScaler
import re
from urllib.parse import unquote
from scipy.stats import entropy
import math
from scipy import ndimage

# Initialize Flask application
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # For local development
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://localhost:5500", "null"],
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST"],
        "supports_credentials": True
    }
})

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application Configuration
app.config.update(
    # Upload settings
    UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads'),  # Absolute path for uploads
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'},  # File type restrictions
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # File size limit (16MB)
    
    # Image quality requirements
    MIN_IMAGE_SIZE=(224, 224),  # Minimum dimensions required (width, height)
    MIN_RESOLUTION_DPI=72,  # Minimum image resolution
    
    # AI Model settings
    MODEL_PATH='ai_model.pkl',  # Path to the trained model
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check UPLOAD_FOLDER path consistency
print(f"Upload folder path: {app.config['UPLOAD_FOLDER']}")
# Should output something like:
# Upload folder path: /your/project/path/static/uploads

# Add route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Add static file handling
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

class AIImageDetector:
    """Implements AI-generated image detection using the trained model"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.class_names = {0: 'AI-generated Image', 1: 'Real Image'}
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the model"""
        try:
            # Check if model file exists first
            model_path = app.config['MODEL_PATH']
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            # Load the trained model
            self.logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            self.logger.info(f"Successfully loaded model of type: {type(self.model)}")
            
            # Initialize scaler
            self.scaler = StandardScaler()
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def preprocess_image(self, image):
        """Preprocess the image for model input"""
        try:
            # Resize and convert to array
            image = image.resize((224, 224))
            img_array = np.array(image)
            
            # Flatten and scale the image
            flattened = img_array.flatten().reshape(1, -1)
            scaled = self.scaler.fit_transform(flattened)
            return scaled
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_image(self, image_path):
        """Predict if an image is AI-generated or real"""
        try:
            # Open and preprocess the image
            image = Image.open(image_path).convert("RGB")
            processed_image = self.preprocess_image(image)
            
            # Get prediction probabilities
            predictions = self.model.predict_proba(processed_image)
            
            # Get probabilities
            ai_probability = predictions[0][0] * 100
            real_probability = predictions[0][1] * 100
            
            # Determine classification based on highest probability
            classification = self.class_names[0] if ai_probability > 50 else self.class_names[1]
            
            return {
                "classification": classification,
                "ai_probability": ai_probability,
                "real_probability": real_probability,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_metadata": self._get_image_metadata(image_path, image),
                "note": "Analysis performed using trained deep learning model.",
                "home_url": url_for('index')
            }
        except Exception as e:
            self.logger.error(f"Error predicting image: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_image_metadata(self, image_path, image):
        """Get basic metadata about the image"""
        return {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "filename": os.path.basename(image_path)
        }

# Initialize AI detector with better error handling
ai_detector = None
try:
    logger.info(f"Attempting to load model from: {app.config['MODEL_PATH']}")
    ai_detector = AIImageDetector()
    logger.info("AI Detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI Detector: {str(e)}")
    logger.error(traceback.format_exc())
    ai_detector = None
    # Try alternative direct loading as fallback
    try:
        logger.info("Attempting direct model loading as fallback")
        model_path = app.config['MODEL_PATH']
        if os.path.exists(model_path):
            # Create a simple model wrapper
            class SimpleModelWrapper:
                def __init__(self):
                    self.model = joblib.load(model_path)
                    self.scaler = StandardScaler()
                    self.class_names = {0: 'AI-generated Image', 1: 'Real Image'}
                    logger.info(f"Simple model wrapper created with model type: {type(self.model)}")
                
                def predict_image(self, image_path):
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize((224, 224))
                    img_array = np.array(image).flatten().reshape(1, -1)
                    scaled = self.scaler.fit_transform(img_array)
                    predictions = self.model.predict_proba(scaled)
                    ai_prob = predictions[0][0] * 100
                    real_prob = predictions[0][1] * 100
                    classification = self.class_names[0] if ai_prob > 50 else self.class_names[1]
                    return {
                        "classification": classification,
                        "ai_probability": ai_prob,
                        "real_probability": real_prob,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "note": "Analysis performed using fallback model loader.",
                        "home_url": url_for('index')
                    }
            
            ai_detector = SimpleModelWrapper()
            logger.info("Fallback model loaded successfully")
    except Exception as fallback_error:
        logger.error(f"Even fallback model loading failed: {str(fallback_error)}")
        logger.error(traceback.format_exc())

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
        return False, "Error validating image quality. Please check if the file is a valid image."

# API Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/result/<filename>')
def show_result(filename):
    """Display detection results for a specific image."""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            response = make_response(render_template('result.html', 
                                error="File not found",
                                details=f"{filename} does not exist",
                                home_url=url_for('index')))  # Add home URL
            response.headers['Cache-Control'] = 'no-store'
            return response

        # Handle cached results
        if request.args.get('results'):
            response = make_response(render_template('result.html',
                                filename=filename,
                                detection_results=request.args.get('results'),
                                home_url=url_for('index')))  # Add home URL
            response.headers['Cache-Control'] = 'no-store'
            return response
        
        # Get AI analysis results
        if ai_detector:
            detection_results = ai_detector.predict_image(filepath)
        else:
            detection_results = {
                "classification": "Error: AI model is unavailable",
                "ai_probability": 0,
                "real_probability": 0,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "AI model is unavailable"
            }
        
        # Get complete server URL for image access
        base_url = request.url_root
        image_url = f"{base_url}uploads/{filename}"
        
        # Create proper response object with home URL
        response = make_response(render_template('result.html',
                                filename=filename,
                                image_url=image_url,
                                detection_results=detection_results,
                                home_url=url_for('index')))  # Add home URL
        
        # Add headers to prevent caching
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
                             
    except Exception as e:
        logger.error(f"Error showing results: {traceback.format_exc()}")
        response = make_response(render_template('result.html',
                            error="Processing error",
                            details=str(e),
                            home_url=url_for('index')))  # Add home URL
        response.headers['Cache-Control'] = 'no-store'
        return response

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload, validation, and detection."""
    try:
        # Check if file was included in request
        if 'file' not in request.files:
            return jsonify({"error": "No file selected. Please select an image to upload."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected. Please select an image to upload."}), 400
        
        # Check file size before loading it
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file position
        
        # Check if file is too large (compare with MAX_CONTENT_LENGTH)
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
            return jsonify({"error": f"File is too large. Maximum allowed size is {max_size_mb:.1f} MB."}), 413
            
        if not file or not allowed_file(file.filename):
            allowed_extensions = ', '.join(app.config['ALLOWED_EXTENSIONS'])
            return jsonify({"error": f"Invalid file type. Only {allowed_extensions} files are allowed."}), 400
            
        # Validate image quality
        is_valid, message = validate_image_quality(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Read the saved file for base64 encoding
            with open(filepath, 'rb') as saved_file:
                image_data = base64.b64encode(saved_file.read()).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            return jsonify({
                "error": "Failed to save uploaded file",
                "details": str(e)
            }), 500

        return jsonify({
            "success": True,
            "redirect": f"/result/{filename}",
            "image_path": f"/uploads/{filename}",
            "image_data": image_data
        }), 200
            
    except Exception as e:
        error_message = str(e)
        
        # Check for common errors and provide user-friendly messages
        if "Request Entity Too Large" in error_message:
            max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
            return jsonify({"error": f"File is too large. Maximum allowed size is {max_size_mb:.1f} MB."}), 413
        
        logger.error(f"Error processing upload: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred while processing your upload. Please try again."}), 500

@app.route('/api/analyze', methods=['POST', 'GET'])
def analyze_image():
    """API endpoint for image analysis"""
    try:
        # Get filename from URL referrer
        filename = None
        referrer = request.headers.get('Referer', '')
        
        if '/result/' in referrer:
            try:
                filename = referrer.split('/result/')[1].split('?')[0]
                logger.info(f"Extracted filename from URL: {filename}")
            except:
                pass
        
        # If no filename in URL, try from form data
        if not filename and 'filename' in request.form:
            filename = request.form.get('filename')
            logger.info(f"Got filename from form: {filename}")
        
        # If still no filename, use most recent upload
        if not filename:
            try:
                uploads_dir = app.config['UPLOAD_FOLDER']
                files = [f for f in os.listdir(uploads_dir) if allowed_file(f)]
                if files:
                    filename = max(files, key=lambda f: os.path.getmtime(os.path.join(uploads_dir, f)))
                    logger.info(f"Using most recent upload: {filename}")
                else:
                    return jsonify({"error": "No files found for analysis"}), 400
            except Exception as e:
                logger.error(f"Error finding recent file: {str(e)}")
                return jsonify({"error": "Could not determine file to analyze"}), 400
        
        # Process with found filename
        clean_filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
        
        if not os.path.exists(filepath):
            logger.error(f"Image file not found: {filepath}")
            return jsonify({"error": "Image file not found"}), 404
        
        # Try to load and use the model, but don't use fallback analysis
        try:
            model_path = app.config['MODEL_PATH']
            
            # Check if model exists
            if not os.path.exists(model_path):
                return jsonify({
                    "error": "AI model is unavailable", 
                    "home_url": url_for('index')
                }), 503
                
            # Load the model
            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            
            # Load and preprocess the image
            image = Image.open(filepath).convert("RGB")
            image = image.resize((224, 224))
            img_array = np.array(image).flatten().reshape(1, -1)
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(img_array)
            
            # Get predictions
            predictions = model.predict_proba(scaled_data)
            
            # Format results
            ai_probability = predictions[0][0] * 100
            real_probability = predictions[0][1] * 100
            classification = "AI-generated Image" if ai_probability > 50 else "Real Image"
            
            # Create result object
            results = {
                "classification": classification,
                "ai_probability": ai_probability,
                "real_probability": real_probability,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_metadata": {
                    "filename": os.path.basename(filepath),
                    "format": image.format,
                    "size": image.size,
                    "mode": image.mode
                },
                "note": "Analysis performed using trained machine learning model.",
                "home_url": url_for('index')
            }
            
            logger.info("Analysis successful with model")
            return jsonify(results), 200
            
        except Exception as model_error:
            # Log the specific error for troubleshooting
            logger.error(f"Model analysis failed: {str(model_error)}")
            logger.error(traceback.format_exc())
            
            # Return error to the client - don't use statistical fallback
            return jsonify({
                "error": "AI model is unavailable",
                "details": str(model_error),
                "home_url": url_for('index')
            }), 503
                
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# Custom error handler for 413 Request Entity Too Large
@app.errorhandler(413)
def request_entity_too_large(error):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return jsonify({"error": f"File is too large. Maximum allowed size is {max_size_mb:.1f} MB."}), 413

# Add alias route for the incorrect /uploads endpoint
@app.route('/uploads', methods=['POST'])
def uploads_alias():
    return upload_image()

# Add startup check
upload_dir = app.config['UPLOAD_FOLDER']
if not os.access(upload_dir, os.W_OK):
    logger.error(f"Write permissions missing for: {upload_dir}")

# Temporary debug route
@app.route('/cors-test')
def cors_test():
    response = jsonify({"message": "CORS test successful"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.after_request
def add_header(response):
    """Add headers to prevent caching and ensure proper navigation."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5500'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/check-model')
def check_model():
    """Debug route to check model status"""
    model_path = app.config['MODEL_PATH']
    response = {
        "model_file_exists": os.path.exists(model_path),
        "model_file_path": os.path.abspath(model_path),
        "model_file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        "ai_detector_initialized": ai_detector is not None,
        "model_loaded": ai_detector is not None and ai_detector.model is not None
    }
    
    if ai_detector and ai_detector.model:
        response["model_type"] = str(type(ai_detector.model))
    
    return jsonify(response)

@app.route('/home')
def go_home():
    """Explicit route to return to the home page"""
    return redirect(url_for('index'))

@app.route('/go-to-home', methods=['GET'])
def navigate_home():
    """Special endpoint that always redirects to home, breaking out of any history issues."""
    response = make_response(redirect(url_for('index')))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Start the Flask development server
    app.run(debug=True, port=5500)