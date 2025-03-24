from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import logging
from PIL import Image
import traceback  # Added for better error tracking
import datetime
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models

# Initialize Flask application
app = Flask(__name__)

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application Configuration
app.config.update(
    # Upload settings
    UPLOAD_FOLDER='static/uploads',  # Upload directory configured
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'},  # File type restrictions
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # File size limit (16MB)
    
    # Image quality requirements
    MIN_IMAGE_SIZE=(224, 224),  # Minimum dimensions required (width, height)
    MIN_RESOLUTION_DPI=72,  # Minimum image resolution
    
    # AI Model settings
    MODEL_CACHE_DIR='models',  # Where to cache AI models
    NOTEBOOK_PATH='aiImage-realImage-classification.ipynb'  # Path to the notebook
)

class AIImageDetector:
    """Implements AI-generated image detection without depending on the notebook"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transform = None
        self.class_names = {0: 'AI-generated Image', 1: 'Real Image'}
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the model"""
        try:
            # Use a pre-trained ResNet model
            self.model = models.resnet50(pretrained=True)
            
            # Modify the final layer for binary classification
            num_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: AI and Real
            
            # Set up the data transformation
            self.data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Move the model to the appropriate device
            self.model = self.model.to(self.device)
            
            # Set the model to evaluation mode
            self.model.eval()
            
            self.logger.info("Successfully initialized the model")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def predict_image(self, image_path):
        """Predict if an image is AI-generated or real"""
        try:
            # Open the image and convert it to RGB
            image = Image.open(image_path).convert("RGB")
            
            # Apply the transformations
            image_tensor = self.data_transform(image).unsqueeze(0)  # Add batch dimension
            
            # Move the image tensor to the correct device (CPU or GPU)
            image_tensor = image_tensor.to(self.device)
            
            # We don't have trained weights, so this is a demonstration
            # For a real implementation, you'd load weights from a trained model
            
            # Create a prediction that simulates AI detection
            # This is just a placeholder since we don't have actual trained weights
            # In a real implementation, you would get predictions from a trained model
            import random
            
            # Simulate AI probability with random value for demonstration
            ai_prob = random.uniform(0, 1)
            
            # Get the probability of the image being AI-generated
            ai_image_prob = ai_prob * 100  # Convert to percentage
            
            # Determine the classification based on the probability
            prediction = self.class_names[0] if ai_image_prob > 50 else self.class_names[1]
            
            return {
                "classification": prediction,
                "ai_probability": ai_image_prob,
                "real_probability": 100 - ai_image_prob,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_metadata": self._get_image_metadata(image_path, image),
                "note": "This is a demonstration using random predictions. In a production environment, this would use a trained model."
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

# Initialize AI detector
try:
    ai_detector = AIImageDetector()
    logger.info("AI Detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI Detector: {str(e)}")
    logger.error(traceback.format_exc())
    ai_detector = None

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
            return jsonify({"error": "File not found"}), 404
            
        # Get AI analysis results
        if ai_detector:
            detection_results = ai_detector.predict_image(filepath)
        else:
            detection_results = {
                "classification": "Error: AI detector not initialized",
                "ai_probability": 0,
                "real_probability": 0,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": "AI detector model failed to load"
            }
        
        return render_template('result.html', 
                             filename=filename, 
                             detection_results=detection_results)
                             
    except Exception as e:
        logger.error(f"Error showing results: {traceback.format_exc()}")
        return jsonify({"error": "Failed to process detection results"}), 500

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
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            return jsonify({"error": "Failed to save uploaded file. Please try again."}), 500
        
        # Redirect to result page
        return jsonify({
            "success": True,
            "redirect": f"/result/{filename}"
        }), 200
            
    except Exception as e:
        error_message = str(e)
        
        # Check for common errors and provide user-friendly messages
        if "Request Entity Too Large" in error_message:
            max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
            return jsonify({"error": f"File is too large. Maximum allowed size is {max_size_mb:.1f} MB."}), 413
        
        logger.error(f"Error processing upload: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred while processing your upload. Please try again."}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for image analysis"""
    try:
        # Check if file was included in request
        if 'file' not in request.files:
            return jsonify({"error": "No file selected. Please select an image to upload."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected. Please select an image to upload."}), 400
            
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file position
        
        # Check if file is too large
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
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            return jsonify({"error": "Failed to save uploaded file. Please try again."}), 500
        
        # Analyze the image
        if ai_detector:
            detection_results = ai_detector.predict_image(filepath)
            return jsonify(detection_results), 200
        else:
            return jsonify({"error": "AI detector not initialized. Please try again later."}), 500
            
    except Exception as e:
        error_message = str(e)
        
        # Check for specific errors
        if "Request Entity Too Large" in error_message:
            max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
            return jsonify({"error": f"File is too large. Maximum allowed size is {max_size_mb:.1f} MB."}), 413
            
        logger.error(f"Error analyzing image: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred while analyzing your image. Please try again."}), 500

# Custom error handler for 413 Request Entity Too Large
@app.errorhandler(413)
def request_entity_too_large(error):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return jsonify({"error": f"File is too large. Maximum allowed size is {max_size_mb:.1f} MB."}), 413

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Start the Flask development server
    app.run(debug=True)