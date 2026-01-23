# Image Based Soft Drink Classification Using a Convolutional Neural Network

## Overview
This project implements an image classification system that identifies different
types of drinks from bottle images using a Convolutional Neural Network (CNN).

The model was developed as part of a Bachelor of Science in Data Science
(neural networks unit).

## Classes
The model classifies the following drink categories:
- Coca-Cola
- Pepsi
- Fanta
- Sprite

## Methodology
- Custom image dataset created and organized by class
- Dataset split into training, validation, and test sets
- Transfer learning using MobileNetV2 (pretrained on ImageNet)
- Data augmentation to improve generalization
- Model evaluation using accuracy and live image prediction

## Tools and Technologies
- Python
- TensorFlow / Keras
- Google Colab
- MobileNetV2
- Matplotlib

## Dataset
The dataset is not included in this repository due to size constraints.
Images were collected manually and organized into class folders.

## Results
The trained model achieves strong performance on unseen images,
with high confidence predictions during live demonstrations.

## How to Run
1. Open the notebook in Google Colab
2. Mount Google Drive
3. Load the saved trained model
4. Run the live image prediction cells

## Author
Charles W.
Bachelor of Science in Data Science  
Neural Networks Unit

