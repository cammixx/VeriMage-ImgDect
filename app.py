# Flask and web framework imports
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response, url_for, redirect, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
# Image processing and ML imports
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
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
    MODEL_PATH='trained_model.pth',  # Path to the trained model
)

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the data transformations
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for progress bar
progress_queue = []

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
            
            # Initialize the model
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            num_classes = 2
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            self.model = self.model.to(device)
            
            # Load the saved model state
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            
            self.logger.info(f"Successfully loaded model of type: {type(self.model)}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
     # Grad-CAM generation function
    def generate_gradcam(self, image_path, alpha=0.5):
    # Load fresh model instance each time
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        num_classes = 2
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device)
        model_path = app.config['MODEL_PATH']
        # Load the saved model state
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        transformed = data_transform(image).unsqueeze(0).to(device)
        transformed.requires_grad_()

        # Variables to store activation and gradients
        global activation, gradients
        activation = None
        gradients = None

        # Hooks
        def save_activation_hook(m, i, o):
            global activation
            activation = o.detach()

        def save_gradient_hook(m, gi, go):
            global gradients
            gradients = go[0].detach()

        # Register hooks
        final_conv = self.model.layer4[1].conv2
        handle1 = final_conv.register_forward_hook(save_activation_hook)
        handle2 = final_conv.register_full_backward_hook(save_gradient_hook)

        try:
            # Forward pass
            output = self.model(transformed)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(output).item()
            pred_label = self.class_names[pred_idx]
            confidence = probs[0][pred_idx].item() * 100
    
            # Backward pass
            self.model.zero_grad()
            output[0, pred_idx].backward()
    
            # Grad-CAM calculation - NO IN-PLACE OPERATIONS!
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
            # Create weighted activations without modifying original tensor
            weighted_activations = activation * pooled_gradients[None, :, None, None]
        
            # Sum along the channel dimension to get heatmap
            heatmap = torch.sum(weighted_activations, dim=1).squeeze()
        
            # Apply ReLU to the heatmap
            heatmap = torch.clamp(heatmap, min=0)
        
            # Convert to numpy and normalize
            heatmap_np = heatmap.cpu().numpy()
            if np.max(heatmap_np) > 0:  # Avoid division by zero
                heatmap_np = heatmap_np / np.max(heatmap_np)
    
            # Create visualization
            plt.figure(figsize=(7, 6))
            ax = plt.subplot(111)
    
            # Convert to RGB for consistency
            img_array = np.array(image)
    
            # Apply colormap to heatmap
            heatmap_resized = np.uint8(255 * cm.jet(heatmap_np)[:, :, :3])
            heatmap_pil = Image.fromarray(heatmap_resized).resize((img_array.shape[1], img_array.shape[0]))
            heatmap_array = np.array(heatmap_pil)
    
            # Blend images
            blended = np.uint8((1 - alpha) * img_array + alpha * heatmap_array)
    
            # Plot
            ax.imshow(blended)
            ax.set_title(f"Prediction: {pred_label} ({confidence:.2f}%)")
            ax.axis('off')
    
            # Add colorbar
            norm = Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Influence Level', rotation=270, labelpad=15)
    
            # Save the visualization
            output_filename = f"gradcam_{os.path.basename(image_path)}"
            output_path = os.path.join(os.path.dirname(image_path), output_filename)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
    
            return {
                'success': True,
                'gradcam_path': output_filename,
                'prediction': pred_label,
                'confidence': confidence
            }

        finally:
            # Remove hooks
            handle1.remove()
            handle2.remove()

    def preprocess_image(self, image):
        """Preprocess the image for model input"""
        try:
            # Apply the transformations
            image_tensor = data_transform(image).unsqueeze(0)
            return image_tensor.to(device)
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_image(self, image_path):
        """Predict if an image is AI-generated or real"""
        try:
            # Open and preprocess the image
            image = Image.open(image_path).convert("RGB")
            processed_image = self.preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(processed_image)
                print("Raw outputs (logits):", outputs)
                probabilities = torch.softmax(outputs, dim=1)
                print("Probabilities after softmax:", probabilities)
                ai_probability = probabilities[0][0].item() * 100
                real_probability = probabilities[0][1].item() * 100
                print("AI PROB:",ai_probability)
                print("REAL PROB:",real_probability)
                _, predicted_class = torch.max(outputs, 1)
                print("Predicted class index:", predicted_class.item())
            
            # Determine classification based on highest probability
            classification = self.class_names[predicted_class.item()]
            
            return {
                "classification": classification,
                "ai_probability": ai_probability,
                "real_probability": real_probability,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_metadata": self._get_image_metadata(image_path, image),
                "note": "Analysis performed using trained PyTorch model.",
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
                    self.model = models.resnet18(weights='IMAGENET1K_V1')
                    num_classes = 2
                    self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
                    self.model = self.model.to(device)
                    self.model.load_state_dict(torch.load(model_path, map_location=device))
                    self.model.eval()
                    logger.info(f"Simple model wrapper created with model type: {type(self.model)}")
                
                def predict_image(self, image_path):
                    image = Image.open(image_path).convert("RGB")
                    processed_image = self.preprocess_image(image)
                    with torch.no_grad():
                        outputs = self.model(processed_image)
                        print("Raw outputs (logits):", outputs)
                        probabilities = torch.softmax(outputs, dim=1)
                        print("Probabilities after softmax:", probabilities)
                        ai_prob = probabilities[0][0].item() * 100
                        real_prob = probabilities[0][1].item() * 100
                        print("AI PROB:",ai_prob)
                        print("REAL PROB:",real_prob)
                        _, predicted_class = torch.max(outputs, 1)
                        print("Predicted class index:", predicted_class.item())
                    classification = "AI-generated Image" if ai_prob > 50 else "Real Image"
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

def delete_previous_images():
    """Delete all previous images in the upload folder"""
    try:
        uploads_dir = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(uploads_dir):
            if allowed_file(filename):
                file_path = os.path.join(uploads_dir, filename)
                os.remove(file_path)
                logger.info(f"Deleted previous image: {filename}")
    except Exception as e:
        logger.error(f"Error deleting previous images: {str(e)}")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/api/grad-cam', methods=['POST'])
def grad_cam_api():
    if 'file' not in request.form:
        return jsonify({'error': 'No file specified'})
    
    filename = request.form['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Get alpha value from request
    alpha = float(request.form.get('alpha', 0.5))
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'})
    
    try:
        # Generate fresh Grad-CAM with current alpha
        result = ai_detector.generate_gradcam(filepath, alpha=alpha)
        
        # Add timestamp to prevent caching
        timestamp = int(time.time())
        
        return jsonify({
            'success': True,
            'gradcam_url': f"/static/uploads/gradcam_{filename}?t={timestamp}",
            'prediction': result.get('prediction', ''),
            'confidence': result.get('confidence', '')
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Detailed error in console
        return jsonify({'error': str(e)})





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
        
        # Delete previous images before saving the new one
        delete_previous_images()
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Update progress to 10% - Starting analysis
            update_progress(0.1)
            
            # Load and preprocess image
            image = Image.open(filepath).convert("RGB")
            update_progress(0.3)  # 30% - Image loaded
            
            # Get model prediction
            if ai_detector:
                update_progress(0.5)  # 50% - Starting model inference
                result = ai_detector.predict_image(filepath)
                update_progress(0.8)  # 80% - Model inference complete
            else:
                raise Exception("AI model is unavailable")
            
            update_progress(1.0)  # 100% - Analysis complete
            
            # Read the saved file for base64 encoding
            with open(filepath, 'rb') as saved_file:
                image_data = base64.b64encode(saved_file.read()).decode('utf-8')
                
            return jsonify({
                "success": True,
                "redirect": f"/result/{filename}",
                "image_path": f"/uploads/{filename}",
                "image_data": image_data
            }), 200
            
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            return jsonify({
                "error": "Failed to save uploaded file",
                "details": str(e)
            }), 500
            
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
            model = models.resnet18(weights='IMAGENET1K_V1')
            num_classes = 2
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model = model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Load and preprocess the image
            image = Image.open(filepath).convert("RGB")
            image = image.resize((224, 224))
            img_tensor = data_transform(image).unsqueeze(0).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                print("Raw outputs (logits):", outputs)
                probabilities = torch.softmax(outputs, dim=1)
                print("Probabilities after softmax:", probabilities)
                ai_probability = probabilities[0][0].item() * 100
                real_probability = probabilities[0][1].item() * 100
                print("AI PROB:",ai_probability)
                print("REAL PROB:",real_probability)
                _, predicted_class = torch.max(outputs, 1)
                print("Predicted class index:", predicted_class.item())
            
            # Format results
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
                "note": "Analysis performed using trained PyTorch model.",
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

# Progress tracking for the analysis animation
@app.route('/progress')
def progress():
    def generate():
        while True:
            if progress_queue:
                progress_value = progress_queue.pop(0)
                data = json.dumps({"progress": progress_value})
                yield f"data: {data}\n\n"
            time.sleep(0.1)
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def update_progress(progress_value):
    """Update the progress value that will be sent to the client"""
    progress_queue.append(progress_value)

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

def cleanup_on_exit():
    """Delete all images in the upload folder on app shutdown."""
    try:
        delete_previous_images()
        logger.info("Cleaned up images on app shutdown.")
    except Exception as e:
        logger.error(f"Error during cleanup on exit: {str(e)}")

# Register the cleanup function to be called on exit
atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Start the Flask development server
    app.run(debug=True, port=5500)