# Fraud_Detection

## Introduction

The **Fraud Detection** project aims to detect AI-generated images and predict the confidence score of the detection. The primary goal of this project is to build a **binary image classification model** capable of accurately distinguishing between **real** and **AI-generated images**. The model utilizes advanced deep learning techniques and is trained to identify subtle patterns that differentiate AI-generated images from real images.

## Dataset

The dataset used for this project is available on [Kaggle: AI-Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images). This dataset contains:

- **539 AI-generated images** (label 0)
- **436 real images** (label 1)

The dataset is split into two main categories, and it is used to train and evaluate the binary image classification model. The images are used to help the model learn to identify the key features that differentiate real images from AI-generated ones.

## Data Preprocessing

In the Fraud Detection project, the following data preprocessing steps are applied to ensure the images are ready for training the model:

### 1. Data Loading & Path Setup

The dataset is loaded directly from Kaggle, which includes two directories:

- `AiArtData`: Contains AI-generated images (label 0).
- `RealArt`: Contains real images (label 1).

The image paths and labels are organized within the project structure to facilitate easy access during training.

### 2. Image Filtering & Labeling

To ensure data consistency, the **make_label()** function is used to:

- Read image files from both the `AiArtData` and `RealArt` directories.
- Filter out unsupported image formats (only **.jpg**, **.jpeg**, and **.png** files are accepted).
- Assign labels: **0** for AI-generated images and **1** for real images.

The image paths and corresponding labels are then stored in a **Pandas DataFrame** for easy manipulation.

### 3. Train-Test Split

To evaluate the model's performance, the dataset is split into:

- **80% Training Set**
- **20% Validation Set**

Stratified sampling is applied to maintain class balance, ensuring that both real and AI-generated images are evenly distributed in both the training and validation sets.

### 4. Image Transformations

Various image transformations are applied to enhance the model's robustness and improve performance:

#### Training Set:
- **Random Resized Crop (224x224)**: Randomly crops the image to a 224x224 size to introduce variability.
- **Random Horizontal Flip**: Randomly flips the image horizontally, improving generalization.
- **Normalization**: Standardizes the image pixel values to a range of 0-1.

#### Validation Set:
- **Resize (256x256)**: Resizes the image to 256x256 pixels to ensure consistent dimensions.
- **Center Crop (224x224)**: Crops the central region of the image to 224x224 pixels.
- **Normalization**: Normalizes the image similar to the training set for consistent input.

### 5. Custom Dataset Class

A custom **PyTorch Dataset** class is created to streamline the data pipeline. This class is responsible for:

- Loading images from disk.
- Applying the appropriate transformations (e.g., resizing, cropping, flipping).
- Returning the image tensor along with the corresponding label for model training.

### 6. Dataloader Setup

To efficiently load data during training, the images are wrapped in **PyTorch DataLoader** objects. These loaders provide:

- **Batch size = 4**: A batch size of 4 is chosen to ensure efficient memory usage during training.
- **Multi-threaded loading (`num_workers=2`)**: Multiple workers are used for parallel data loading, ensuring faster processing of images.

By setting up these preprocessing and data loading steps, we ensure that the model receives well-processed, consistent input data, which is crucial for achieving high performance in AI-generated image detection.

