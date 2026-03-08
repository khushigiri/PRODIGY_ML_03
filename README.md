# Cats vs Dogs Image Classification using SVM

## Project Overview

This project implements a Support Vector Machine (SVM) based image classification model to distinguish between cats and dogs using images from the Kaggle Cats vs Dogs dataset.

The images are preprocessed and transformed into meaningful feature representations using Histogram of Oriented Gradients (HOG). These features are then used to train an SVM classifier to predict whether an image contains a cat or a dog.

This project demonstrates the application of machine learning techniques in image classification using feature extraction and supervised learning.

---

## Objectives

* Load and preprocess image data
* Extract meaningful features using HOG
* Train an SVM classifier
* Evaluate model performance
* Predict the class of new images

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Scikit-image (HOG feature extraction)

---

## Dataset

The project uses the Cats vs Dogs dataset from Kaggle.

---

## Workflow

### Image Preprocessing

* Images are loaded using OpenCV
* Resized to 64 × 64 pixels
* Converted to grayscale to simplify computation

### Feature Extraction

The Histogram of Oriented Gradients (HOG) technique is used to extract important visual features such as:

* edges
* gradients
* shapes

This converts images into numerical feature vectors suitable for machine learning models.

### Model Training

A Linear Support Vector Machine (LinearSVC) is trained on the extracted HOG features.

### Model Evaluation

The trained model is tested on unseen images from the test dataset to compute classification accuracy.

### Prediction

The trained model can predict whether a new image contains a cat or a dog.

---

## Results

Using HOG feature extraction with SVM, the model achieves approximately:

Accuracy: ~80–85%

This approach significantly improves performance compared to training SVM directly on raw image pixels.

---
