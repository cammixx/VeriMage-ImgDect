# Fraud_Detection_Project

## Introduction

The **Fraud Detection** project aims to detect AI-generated images and predict the confidence score of the detection. The primary goal of this project is to build a **binary image classification model** capable of accurately distinguishing between **real** and **AI-generated images**. The model utilizes advanced deep learning techniques and is trained to identify subtle patterns that differentiate AI-generated images from real images.

```
FRAUD_DETECTION/
├── static/
│   ├── uploads/          # Store uploaded images
│   ├── styles.css        # CSS styles for the application
│   └── script.js         # JavaScript for frontend functionality
├── templates/           
│   ├── index.html        # Upload image interface
│   └── result.html       # Results display page
├── dataset/             # Training and testing datasets
├── aiImage-realImage-classification.ipynb  # Reference AI detection model notebook
├── app.py               # Flask application with AI model implementation
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Implementation Approach

This project provides a web application for AI-generated image detection with the following components:

1. **Pretrained Model Framework**: The application uses a ResNet-50 architecture as the foundation for the image classification model. In a production environment, this would be fully trained and fine-tuned.

2. **Demonstration Mode**: Currently, the system operates in demonstration mode, generating simulated predictions to showcase the user interface and functionality without requiring complex model training.

3. **Web Interface**: A clean and responsive web interface allows users to upload images and view detection results, including confidence scores and image metadata.

4. **Future Integration Path**: The project is structured to easily integrate a fully trained model, either from the reference notebook or from a dedicated model training pipeline.

## Usage

1. Access the web interface through your browser
2. Upload an image for analysis (supported formats: JPG, JPEG, PNG)
3. View detailed results including:
   - Classification (AI-generated or Real image)
   - Confidence score
   - Image metadata

## Installation and Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Access the application in your browser at `http://localhost:5500`

## Model Details

The demonstration model is based on a pre-trained ResNet architecture. Key features:

- Uses a ResNet18 architecture 
- Binary classification (AI-generated vs Real)
- Returns simulated probability scores
- Classifies the image based on probability scores

## Dataset

The dataset used for this project is available on [Kaggle: AI-Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images). This dataset contains:

- **539 AI-generated images** (label 0)
- **436 real images** (label 1)

The dataset is split into two main categories, and it is used to train and evaluate the binary image classification model. The images are used to help the model learn to identify the key features that differentiate real images from AI-generated ones.

## Data Preprocessing

The following data preprocessing steps are applied to ensure the images are ready for analysis:

### Image Transformations

Various image transformations are applied to enhance the model's robustness and improve performance:

#### Training Set:
- **Random Resized Crop (224x224)**: Randomly crops the image to a 224x224 size to introduce variability.
- **Random Horizontal Flip**: Randomly flips the image horizontally, improving generalization.
- **Normalization**: Standardizes the image pixel values.

#### Validation & Testing:
- **Resize (256x256)**: Resizes the image to 256x256 pixels to ensure consistent dimensions.
- **Center Crop (224x224)**: Crops the central region of the image to 224x224 pixels.
- **Normalization**: Normalizes the image for consistent input.

## Technical Requirements

- Python 3.8+
- PyTorch 1.12.0+
- Flask 2.0.0+
- Pillow 9.0.0+
