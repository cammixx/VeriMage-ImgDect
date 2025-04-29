# Flask and web framework imports
import atexit, os, time, json, base64, logging, traceback, datetime, psutil
from collections import deque
from flask import Flask, request, jsonify, render_template, url_for, redirect, Response, stream_with_context, make_response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Image processing and ML imports
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ----------------- CONFIGURATION -----------------

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App settings
app.config.update(
    SEND_FILE_MAX_AGE_DEFAULT=0,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads'),
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg'},
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB
    MIN_IMAGE_SIZE=(224, 224),
    MIN_RESOLUTION_DPI=72,
    MODEL_PATH='trained_model.pth',
)

# CORS setup
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://localhost:5500", "null"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "methods": ["GET", "POST", "OPTIONS"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "X-Progress-ID"]
    }
})

# Setup ML processing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Progress tracking
progress_queues = {}

# ----------------- HELPER FUNCTIONS -----------------

def get_client_id():
    """Generate unique client identifier"""
    return f"{request.remote_addr}_{time.time()}"

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image_quality(file_stream):
    """Validate image dimensions and resolution"""
    try:
        image = Image.open(file_stream)
        width, height = image.size
        dpi = image.info.get('dpi', (0, 0))[0]
        
        if width < app.config['MIN_IMAGE_SIZE'][0] or height < app.config['MIN_IMAGE_SIZE'][1]:
            return False, f"Image too small. Minimum size is {app.config['MIN_IMAGE_SIZE'][0]}x{app.config['MIN_IMAGE_SIZE'][1]} pixels"
        
        if dpi and dpi < app.config['MIN_RESOLUTION_DPI']:
            return False, f"Image resolution too low. Minimum is {app.config['MIN_RESOLUTION_DPI']} DPI"
        
        file_stream.seek(0)
        return True, "Image meets quality requirements"
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False, "Error validating image quality. Please check if the file is a valid image."

def delete_previous_images():
    """Delete all previous images in the upload folder"""
    try:
        uploads_dir = app.config['UPLOAD_FOLDER']
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if allowed_file(filename):
                    file_path = os.path.join(uploads_dir, filename)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted previous image: {filename}")
                    except OSError:
                        logger.error(f"Failed to delete: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting previous images: {str(e)}")

def update_progress(progress_value):
    """Update progress for all connected clients"""
    current_queues = list(progress_queues.keys())
    for client_id in current_queues:
        if client_id in progress_queues:
            progress_queues[client_id].append(progress_value)

def cleanup_on_exit():
    """Delete all images in the upload folder on app shutdown"""
    try:
        delete_previous_images()
        logger.info("Cleaned up images on app shutdown.")
    except Exception as e:
        logger.error(f"Error during cleanup on exit: {str(e)}")

# ----------------- AI MODEL CLASS -----------------

class AIImageDetector:
    """Implements AI-generated image detection using PyTorch"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.class_names = {0: 'AI-generated Image', 1: 'Real Image'}
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the model"""
        try:
            model_path = app.config['MODEL_PATH']
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            num_classes = 2
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            self.model = self.model.to(device)
            
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            
            self.logger.info(f"Successfully loaded model of type: {type(self.model)}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def generate_gradcam(self, image_path, alpha=0.5):
        """Generate Grad-CAM visualization for the image"""
        try:
            # Ensure the image exists
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return {'error': f"Image file not found: {os.path.basename(image_path)}"}
            
            # Prepare output path first to avoid file issues later
            output_dir = os.path.dirname(image_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Include alpha in filename to generate different versions for different alpha values
            basename = os.path.basename(image_path)
            filename_parts = os.path.splitext(basename)
            output_filename = f"gradcam_{filename_parts[0]}_alpha{alpha:.1f}{filename_parts[1]}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Check if file already exists with the same alpha - skip processing if so
            if os.path.exists(output_path):
                self.logger.info(f"GradCAM visualization already exists with alpha={alpha}: {output_path}")
                return {
                    'success': True,
                    'gradcam_path': output_filename,
                    'prediction': "Re-using existing visualization",
                    'confidence': 100.0,
                    'alpha': alpha
                }
            
            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
                self.logger.info(f"Successfully loaded image: {image_path}")
            except Exception as e:
                self.logger.error(f"Error loading image: {str(e)}")
                return {'error': f"Error loading image: {str(e)}"}
                
            # Load model in a controlled way
            try:
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                # Use CPU for stability
                local_device = torch.device("cpu")
                
                # Load fresh model instance
                self.model = models.resnet18(weights='IMAGENET1K_V1')
                num_classes = 2
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
                self.model = self.model.to(local_device)
                self.model.load_state_dict(torch.load(app.config['MODEL_PATH'], map_location=local_device))
                self.model.eval()
                self.logger.info("Successfully loaded model for GradCAM")
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                return {'error': f"Error loading model: {str(e)}"}

            # Basic prediction first (without GradCAM) to check if model works
            try:
                # Transform image
                transformed = data_transform(image).unsqueeze(0).to(local_device)
                
                # Run basic prediction
                with torch.no_grad():
                    output = self.model(transformed)
                    probs = torch.softmax(output, dim=1)
                    pred_idx = torch.argmax(output).item()
                    pred_label = self.class_names[pred_idx]
                    confidence = probs[0][pred_idx].item() * 100
                
                self.logger.info(f"Basic prediction successful: {pred_label} ({confidence:.2f}%)")
            except Exception as e:
                self.logger.error(f"Error during basic prediction: {str(e)}")
                return {'error': f"Error during prediction: {str(e)}"}
            
            # Now try GradCAM computation in isolated sections
            try:
                # Setup for gradients
                transformed = data_transform(image).unsqueeze(0).to(local_device)
                transformed.requires_grad_()
                
                # Variables for hooks
                global activation, gradients
                activation = None
                gradients = None

                # Hook functions
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
                
                # Forward pass
                output = self.model(transformed)
                
                # Backward pass
                self.model.zero_grad()
                output[0, pred_idx].backward()
                
                # Remove hooks immediately after use
                handle1.remove()
                handle2.remove()
                
                # Check if hooks worked
                if activation is None or gradients is None:
                    self.logger.error("Hooks failed to capture activation or gradients")
                    return {'error': "Failed to compute model gradients"}
                
                # GradCAM calculation
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                weighted_activations = activation * pooled_gradients[None, :, None, None]
                heatmap = torch.sum(weighted_activations, dim=1).squeeze()
                heatmap = torch.clamp(heatmap, min=0)
                
                # Process heatmap
                heatmap_np = heatmap.cpu().numpy()
                if np.max(heatmap_np) > 0:
                    heatmap_np = heatmap_np / np.max(heatmap_np)
                
                self.logger.info("GradCAM computation successful")
            except Exception as e:
                self.logger.error(f"Error during GradCAM computation: {str(e)}")
                return {'error': f"Error computing heatmap: {str(e)}"}
            
            # Visualization and saving with matplotlib in isolated section
            try:
                # Use non-interactive backend
                import matplotlib
                matplotlib.use('Agg')
                
                # Create visualization
                plt.figure(figsize=(7, 6))
                ax = plt.subplot(111)
                img_array = np.array(image)
                
                # Apply colormap
                heatmap_resized = np.uint8(255 * cm.jet(heatmap_np)[:, :, :3])
                heatmap_pil = Image.fromarray(heatmap_resized).resize((img_array.shape[1], img_array.shape[0]))
                heatmap_array = np.array(heatmap_pil)
                
                # Blend images with the specified alpha value
                blended = np.uint8((1 - alpha) * img_array + alpha * heatmap_array)
                
                # Plot and save
                ax.imshow(blended)
                ax.set_title(f"Prediction: {pred_label} ({confidence:.2f}%)")
                ax.axis('off')
                
                # Add colorbar
                norm = Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Influence Level', rotation=270, labelpad=15)
                
                # Save visualization with error handling
                try:
                    plt.savefig(output_path, bbox_inches='tight', dpi=100)
                    plt.close('all')  # Close all figures to prevent memory leaks
                except Exception as save_error:
                    self.logger.error(f"Error saving figure: {str(save_error)}")
                    # Try with a simpler approach if the first fails
                    try:
                        plt.savefig(output_path)
                        plt.close('all')
                    except:
                        self.logger.error("Failed with simple save too")
                        return {'error': f"Failed to save visualization: {str(save_error)}"}
                
                # Verify file was created
                if not os.path.exists(output_path):
                    self.logger.error(f"Failed to save GradCAM visualization: {output_path}")
                    return {'error': "Failed to save visualization"}
                
                self.logger.info(f"Successfully saved GradCAM with alpha={alpha} to: {output_path}")
                
                # Force garbage collection again
                gc.collect()
                
                return {
                    'success': True,
                    'gradcam_path': output_filename,
                    'prediction': pred_label,
                    'confidence': confidence,
                    'alpha': alpha
                }
            except Exception as e:
                self.logger.error(f"Error during visualization: {str(e)}")
                return {'error': f"Error creating visualization: {str(e)}"}
                
        except Exception as e:
            self.logger.error(f"Error generating GradCAM: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': f"Error generating visualization: {str(e)}"}

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        try:
            image_tensor = data_transform(image).unsqueeze(0)
            return image_tensor.to(device)
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_image(self, image_path):
        """Predict if an image is AI-generated or real"""
        try:
            # Process image
            image = Image.open(image_path).convert("RGB")
            processed_image = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(processed_image)
                probabilities = torch.softmax(outputs, dim=1)
                ai_probability = probabilities[0][0].item() * 100
                real_probability = probabilities[0][1].item() * 100
                _, predicted_class = torch.max(outputs, 1)
            
            # Format results
            classification = self.class_names[predicted_class.item()]
            
            return {
                "classification": classification,
                "ai_probability": ai_probability,
                "real_probability": real_probability,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_metadata": {
                    "format": image.format,
                    "size": image.size,
                    "mode": image.mode,
                    "filename": os.path.basename(image_path)
                },
                "note": "Analysis performed using trained PyTorch model.",
                "home_url": url_for('index')
            }
        except Exception as e:
            self.logger.error(f"Error predicting image: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

# ----------------- INITIALIZE MODEL -----------------

# Initialize AI detector with error handling
ai_detector = None
try:
    logger.info(f"Loading model from: {app.config['MODEL_PATH']}")
    ai_detector = AIImageDetector()
    logger.info("AI Detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI Detector: {str(e)}")
    logger.error(traceback.format_exc())
    ai_detector = None

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------- MAIN ROUTES -----------------

@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and analysis"""
    try:
        # Initialize client progress tracking
        client_id = get_client_id()
        progress_queues[client_id] = deque()
        
        # Validate file submission
        if 'file' not in request.files:
            return jsonify({"error": "No file selected. Please select an image."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected. Please select an image."}), 400
        
        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
            return jsonify({"error": f"File too large. Maximum size is {max_size_mb:.1f} MB."}), 413
        
        # Validate file type and quality
        if not allowed_file(file.filename):
            allowed_extensions = ', '.join(app.config['ALLOWED_EXTENSIONS'])
            return jsonify({"error": f"Invalid file type. Only {allowed_extensions} allowed."}), 400
        
        is_valid, message = validate_image_quality(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Save file and process
        try:
            # Prepare directory
            upload_dir = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_dir, mode=0o755, exist_ok=True)
            
            # Delete previous images
            delete_previous_images()
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            
            # Verify file was saved
            if not os.path.exists(filepath):
                logger.error(f"File save verification failed: {filepath}")
                return jsonify({"error": "Server couldn't save the file"}), 500
            
            # Process image with AI model
            update_progress(0.1)  # Starting analysis
            image = Image.open(filepath).convert("RGB")
            update_progress(0.3)  # Image loaded
            
            if ai_detector:
                update_progress(0.5)  # Starting inference
                result = ai_detector.predict_image(filepath)
                update_progress(0.8)  # Inference complete
            else:
                raise Exception("AI model is unavailable")
            
            update_progress(1.0)  # Analysis complete
            
            # Prepare response
            with open(filepath, 'rb') as saved_file:
                image_data = base64.b64encode(saved_file.read()).decode('utf-8')
            
            return jsonify({
                "success": True,
                "redirect": f"/result/{filename}",
                "image_path": f"/uploads/{filename}",
                "image_data": image_data
            }), 200
            
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            logger.error(f"Path: {filepath}, Dir exists: {os.path.exists(os.path.dirname(filepath))}")
            return jsonify({"error": "File processing failed", "details": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Upload error: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred during upload."}), 500

@app.route('/result/<filename>')
def show_result(filename):
    """Display detection results for an image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return render_template('result.html', 
                                error="File not found",
                                details=f"{filename} does not exist",
                                home_url=url_for('index')), 404
        
        # Use cached results if available
        if request.args.get('results'):
            return render_template('result.html',
                                filename=filename,
                                detection_results=request.args.get('results'),
                                home_url=url_for('index'))
        
        # Get fresh analysis
        detection_results = {
            "classification": "Error: AI model is unavailable",
            "ai_probability": 0,
            "real_probability": 0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": "AI model is unavailable"
        }
        
        if ai_detector:
            detection_results = ai_detector.predict_image(filepath)
        
        # Complete URL for image access
        base_url = request.url_root
        image_url = f"{base_url}uploads/{filename}"
        
        return render_template('result.html',
                            filename=filename,
                            image_url=image_url,
                            detection_results=detection_results,
                            home_url=url_for('index'))
                            
    except Exception as e:
        logger.error(f"Error showing results: {traceback.format_exc()}")
        return render_template('result.html',
                        error="Processing error",
                        details=str(e),
                        home_url=url_for('index')), 500

@app.route('/api/analyze', methods=['POST', 'GET'])
def analyze_image():
    """API endpoint for image analysis"""
    try:
        # Get filename from various sources
        filename = None
        
        # From URL referrer
        referrer = request.headers.get('Referer', '')
        if '/result/' in referrer:
            try:
                filename = referrer.split('/result/')[1].split('?')[0]
            except:
                pass
        
        # From form data
        if not filename and 'filename' in request.form:
            filename = request.form.get('filename')
        
        # From most recent upload
        if not filename:
            uploads_dir = app.config['UPLOAD_FOLDER']
            files = [f for f in os.listdir(uploads_dir) if allowed_file(f)]
            if files:
                filename = max(files, key=lambda f: os.path.getmtime(os.path.join(uploads_dir, f)))
            else:
                return jsonify({"error": "No files found for analysis"}), 400
        
        # Process the image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({"error": "Image file not found"}), 404
        
        # Run model inference
        model_path = app.config['MODEL_PATH']
        if not os.path.exists(model_path):
            return jsonify({"error": "AI model is unavailable"}), 503
            
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_classes = 2
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Process image and get prediction
        image = Image.open(filepath).convert("RGB")
        image = image.resize((224, 224))
        img_tensor = data_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            ai_probability = probabilities[0][0].item() * 100
            real_probability = probabilities[0][1].item() * 100
            _, predicted_class = torch.max(outputs, 1)
        
        # Format results
        classification = "AI-generated Image" if ai_probability > 50 else "Real Image"
        
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
        
        return jsonify(results), 200
            
    except Exception as e:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/analyze/<filename>')
def analyze_more(filename):
    """Handle requests for more analysis on an existing image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            response = jsonify({"error": "Image not found"})
            response.headers.update({
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                'Content-Type': 'application/json'
            })
            return response, 404
            
        # Generate additional analysis if AI detector is available
        if ai_detector:
            # Get basic results
            results = ai_detector.predict_image(filepath)
            
            # Add extended analysis data
            results["extended_analysis"] = True
            results["confidence_level"] = "High" if abs(results["ai_probability"] - results["real_probability"]) > 40 else "Medium"
            
            # Additional key the frontend might expect
            results["more_details_available"] = True
            
            response = jsonify(results)
            response.headers.update({
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                'Content-Type': 'application/json'
            })
            return response, 200
        else:
            response = jsonify({"error": "AI model unavailable"})
            response.headers.update({
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                'Content-Type': 'application/json'
            })
            return response, 503
            
    except Exception as e:
        logger.error(f"Error in additional analysis: {str(e)}")
        logger.error(traceback.format_exc())
        response = jsonify({"error": "Analysis failed", "details": str(e)})
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Content-Type': 'application/json'
        })
        return response, 500

@app.route('/api/grad-cam', methods=['POST'])
def grad_cam_api():
    """Generate and return Grad-CAM visualization"""
    try:
        # Log system info for debugging
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        logger.info(f"Python version: {sys.version}")
        
        filename = None
        
        # Check if file is in the request files
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Saved uploaded file to: {filepath}")
        # Check if file is in form data as a string (filename)
        elif 'file' in request.form and request.form['file'] != '':
            filename = secure_filename(request.form['file'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Verify the file exists
            if not os.path.exists(filepath):
                logger.error(f"File not found for GradCAM: {filepath}")
                return jsonify({'error': f'File not found: {filename}'}), 404
                
            logger.info(f"Using existing file: {filepath}")
        else:
            logger.error("No file specified in request")
            return jsonify({'error': 'No file specified'}), 400
        
        # Check if app can write to the directory
        upload_dir = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir):
            try:
                os.makedirs(upload_dir, exist_ok=True)
                logger.info(f"Created upload directory: {upload_dir}")
            except Exception as dir_error:
                logger.error(f"Failed to create upload directory: {str(dir_error)}")
                return jsonify({'error': f'Server directory error: {str(dir_error)}'}), 500
        
        if not os.access(upload_dir, os.W_OK):
            logger.error(f"Cannot write to directory: {upload_dir}")
            return jsonify({'error': 'Server cannot write to upload directory'}), 500
        
        # Log the file details to help with debugging
        logger.info(f"Generating GradCAM for: {filepath} (exists: {os.path.exists(filepath)}, size: {os.path.getsize(filepath) if os.path.exists(filepath) else 0} bytes)")
        
        # Verify AI detector is available
        if not ai_detector:
            logger.error("AI detector not available for GradCAM generation")
            return jsonify({'error': 'AI model is unavailable'}), 503
        
        # Parse alpha parameter with proper error handling
        try:
            # Get alpha from form data and convert to float
            alpha_str = request.form.get('alpha', '0.5')
            alpha = float(alpha_str)
            # Clamp alpha to valid range
            alpha = max(0.1, min(0.9, alpha))
            logger.info(f"Using alpha value: {alpha} (from input: {alpha_str})")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid alpha value: {request.form.get('alpha')} - {str(e)}")
            # Use default if conversion fails
            alpha = 0.5
            logger.info(f"Using default alpha value: {alpha}")
        
        # Generate the GradCAM visualization
        try:
            logger.info("Starting GradCAM generation")
            result = ai_detector.generate_gradcam(filepath, alpha=alpha)
            logger.info(f"GradCAM generation completed with result: {result}")
        except Exception as grad_error:
            logger.error(f"Unhandled error in generate_gradcam: {str(grad_error)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Server error generating visualization: {str(grad_error)}'}), 500
        
        # Check for errors in the result
        if 'error' in result:
            logger.error(f"GradCAM generation returned error: {result['error']}")
            return jsonify({'error': result['error']}), 500
            
        if not result.get('success', False):
            logger.error("GradCAM generation failed without specific error")
            return jsonify({'error': 'Visualization generation failed'}), 500
        
        # Success path
        timestamp = int(time.time())
        gradcam_path = f"/static/uploads/{result['gradcam_path']}?t={timestamp}"
        
        # Verify the GradCAM file was created
        gradcam_filename = result['gradcam_path']
        gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        
        if not os.path.exists(gradcam_filepath):
            logger.error(f"GradCAM file not found after generation: {gradcam_filepath}")
            return jsonify({'error': 'Visualization file not created'}), 500
            
        logger.info(f"GradCAM generated successfully: {gradcam_filepath} (size: {os.path.getsize(gradcam_filepath)} bytes)")
        
        return jsonify({
            'success': True,
            'gradcam_url': gradcam_path,
            'prediction': result.get('prediction', ''),
            'confidence': result.get('confidence', ''),
            'alpha': alpha  # Return the actual alpha used
        }), 200
    
    except Exception as e:
        logger.error(f"GradCAM API error: {traceback.format_exc()}")
        return jsonify({'error': f"Error processing request: {str(e)}"}), 500

@app.route('/progress')
def progress():
    """Stream progress updates to client"""
    def generate(client_id):
        try:
            while True:
                if client_id in progress_queues and progress_queues[client_id]:
                    progress_value = progress_queues[client_id].popleft()
                    data = json.dumps({"progress": progress_value})
                    yield f"data: {data}\n\n"
                time.sleep(0.5)
        except GeneratorExit:
            if client_id in progress_queues:
                del progress_queues[client_id]

    client_id = get_client_id()
    progress_queues[client_id] = deque()
    response = Response(stream_with_context(generate(client_id)), mimetype='text/event-stream')
    return response

# ----------------- UTILITY ROUTES -----------------

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/go-home')
def go_home():
    """Redirect to home page"""
    return redirect(url_for('index'))

@app.route('/go-to-home', methods=['GET'])
def navigate_home():
    """Break out of history issues and go to home"""
    response = make_response(redirect(url_for('index')))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
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

@app.route('/debug/upload-dir')
def debug_upload_dir():
    """Debug route to check upload directory"""
    dir_path = app.config['UPLOAD_FOLDER']
    exists = os.path.exists(dir_path)
    
    return jsonify({
        "upload_folder": dir_path,
        "exists": exists,
        "writable": os.access(dir_path, os.W_OK) if exists else False,
        "files": os.listdir(dir_path) if exists else [],
        "disk_usage": psutil.disk_usage(dir_path)._asdict() if exists else {}
    })

@app.route('/static/uploads/<path:filename>')
def serve_static_upload(filename):
    """Serve static files from the upload directory"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ----------------- ERROR HANDLERS -----------------

@app.errorhandler(413)
def request_entity_too_large(error):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return jsonify({"error": f"File is too large. Maximum allowed size is {max_size_mb:.1f} MB."}), 413

@app.after_request
def add_header(response):
    """Add headers to prevent caching for dynamic content only"""
    # Only apply no-cache headers to HTML responses and API endpoints
    if response.mimetype == 'text/html' or '/api/' in request.path:
        response.headers.update({
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0'
        })
    # Allow browser caching for static resources
    elif '/static/' in request.path or '/uploads/' in request.path:
        # Don't modify existing cache headers for static content
        pass
    
    # Ensure no redirects are added for regular page loads
    if response.status_code == 200 and request.method == 'GET':
        # Make sure we're not redirecting to the index page
        pass
        
    return response

# ----------------- APP INITIALIZATION -----------------

# Register the cleanup function
atexit.register(cleanup_on_exit)

# Add a 404 handler to prevent falling back to index page
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors with a custom page"""
    return render_template('404.html', error=str(e)), 404 if os.path.exists('templates/404.html') else jsonify({"error": "Page not found", "details": str(e)}), 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5500)