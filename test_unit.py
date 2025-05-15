import unittest
import os
import tempfile
from PIL import Image
import numpy as np
import torch
from app import app, AIImageDetector, allowed_file, validate_image_quality

class TestFraudDetection(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Configure Flask for testing
        app.config.update(
            TESTING=True,
            SERVER_NAME='localhost',
            APPLICATION_ROOT='/',
            PREFERRED_URL_SCHEME='http'
        )
        
        self.app = app.test_client()
        self.app.testing = True
        self.test_upload_folder = tempfile.mkdtemp()
        app.config['UPLOAD_FOLDER'] = self.test_upload_folder
        
        # Create a test image
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.test_image_path = os.path.join(self.test_upload_folder, 'test.jpg')
        self.test_image.save(self.test_image_path)

    def tearDown(self):
        """Clean up after tests"""
        for file in os.listdir(self.test_upload_folder):
            os.remove(os.path.join(self.test_upload_folder, file))
        os.rmdir(self.test_upload_folder)

    def test_allowed_file(self):
        """Test file extension validation"""
        self.assertTrue(allowed_file('test.jpg'))
        self.assertTrue(allowed_file('test.jpeg'))
        self.assertTrue(allowed_file('test.png'))
        self.assertFalse(allowed_file('test.gif'))
        self.assertFalse(allowed_file('test.pdf'))

    def test_validate_image_quality(self):
        """Test image quality validation"""
        # Test valid image
        with open(self.test_image_path, 'rb') as f:
            is_valid, message = validate_image_quality(f)
            self.assertTrue(is_valid)
            self.assertEqual(message, "Image meets quality requirements")

        # Test small image
        small_image = Image.new('RGB', (100, 100), color='red')
        small_image_path = os.path.join(self.test_upload_folder, 'small.jpg')
        small_image.save(small_image_path)
        with open(small_image_path, 'rb') as f:
            is_valid, message = validate_image_quality(f)
            self.assertFalse(is_valid)
            self.assertIn("Image too small", message)

    def test_ai_detector_initialization(self):
        """Test AI detector initialization"""
        detector = AIImageDetector()
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.class_names[0], 'AI-generated Image')
        self.assertEqual(detector.class_names[1], 'Real Image')

    def test_preprocess_image(self):
        """Test image preprocessing"""
        detector = AIImageDetector()
        processed = detector.preprocess_image(self.test_image)
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape[0], 1)  # Batch size
        self.assertEqual(processed.shape[1], 3)  # RGB channels

    def test_predict_image(self):
        """Test image prediction"""
        detector = AIImageDetector()
        with app.app_context():
            result = detector.predict_image(self.test_image_path)
            
            self.assertIsNotNone(result)
            self.assertIn('classification', result)
            self.assertIn('ai_probability', result)
            self.assertIn('real_probability', result)
            self.assertIn('timestamp', result)
            self.assertIn('image_metadata', result)
            
            # Check probabilities sum to approximately 100%
            total_prob = result['ai_probability'] + result['real_probability']
            self.assertAlmostEqual(total_prob, 100.0, delta=0.1)

    def test_generate_gradcam(self):
        """Test GradCAM generation"""
        detector = AIImageDetector()
        with app.app_context():
            result = detector.generate_gradcam(self.test_image_path)
            
            self.assertIsNotNone(result)
            if 'error' not in result:
                self.assertIn('gradcam_path', result)
                self.assertIn('prediction', result)
                self.assertIn('confidence', result)
                self.assertIn('alpha', result)
                
                # Verify GradCAM file was created
                gradcam_path = os.path.join(self.test_upload_folder, result['gradcam_path'])
                self.assertTrue(os.path.exists(gradcam_path))

if __name__ == '__main__':
    unittest.main() 