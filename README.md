# Fraud_Detection
The Fraud Detection project focuses on identifying AI-generated images and estimating the confidence score of the detection. Its primary goal is to build a binary image classification model that accurately differentiates between real and AI-generated images.

## Dataset
<!-- Dataset details to be updated-->

## Data Preprocessing
In your AI Image Detection project, the data preprocessing steps are as follows:

### 1. Data Loading & Path Setup

The dataset is loaded from Kaggle, with two directories:
AiArtData: AI-generated images (Label 0)
RealArt: Real images (Label 1)

### 2. Image Filtering & Labeling

The make_label() function: Reads image files from both directories.
Filters out unsupported image formats (only .jpg, .jpeg, .png are allowed).
Assigns labels (0 for AI, 1 for Real).

Stores image paths and labels in a Pandas DataFrame.

### 3. Train-Test Split

The dataset is split into:
80% Training Set
20% Validation Set
Stratified sampling is applied to maintain class balance.

### 4. Image TransformationsDifferent image augmentations are applied:

Training Set:
Random Resized Crop (224x224)
Random Horizontal Flip
Normalization
Validation Set:
Resize (256x256)
Center Crop (224x224)
Normalization

### 5. Custom Dataset Class

A custom PyTorch Dataset class:

Loads images.
Applies the transformations.
Returns the image tensor and corresponding label.

### 6. Dataloader Setup

Data is wrapped into Dataloader objects with:

Batch size = 4
Multi-threaded loading (num_workers=2)




