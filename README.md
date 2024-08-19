# Retinal Disease Classification using Transfer Learning

This project aims to classify retinal diseases, including cataract, diabetic retinopathy, glaucoma, and normal retina, using deep learning models. The models used include VGG16, ResNet152V2, and EfficientNetB7, all fine-tuned with pre-trained weights from ImageNet.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Project Overview

The objective of this project is to classify retinal images into four categories: cataract, diabetic retinopathy, glaucoma, and normal. The models are built using TensorFlow and Keras, employing transfer learning from pre-trained models (VGG16, ResNet152V2, and EfficientNetB7).

## Dataset

- The dataset consists of images from three classes: cataract, diabetic retinopathy, glaucoma, and normal.
- Data is divided into training, validation, and testing sets:
  - **Training Set**: 2108 images
  - **Validation Set**: 1053 images
  - **Test Set**: 1056 images

## Model Architecture

Three models are used for classification:
- **VGG16**
- **ResNet152V2**
- **EfficientNetB7**

### Fine-tuning Process:
- All layers in the base models are frozen to retain the learned features from ImageNet.
- A global average pooling layer is added, followed by a dense layer with 1024 units and ReLU activation.
- The final layer is a dense layer with softmax activation for multi-class classification.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/retinal-disease-classification.git
   cd retinal-disease-classification
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset:**
   - Organize the dataset into `Train`, `Valid`, and `Test` directories, each containing subfolders for each class (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`).

2. **Run the Training Script:**
   - Execute the script to train the models:

     ```bash
     python train.py
     ```

3. **Evaluate the Models:**
   - After training, evaluate the models on the test set:

     ```bash
     python evaluate.py
     ```

## Results

The models achieved the following performance on the test set:

- **VGG16:**
  - Accuracy: 73%
  - Precision, Recall, F1-Score for each class: 
                       precision    recall  f1-score   support

            cataract       0.70      0.91      0.79       260
diabetic_retinopathy       0.94      0.77      0.85       275
            glaucoma       0.76      0.39      0.51       252
              normal       0.62      0.84      0.72       269

            accuracy                           0.73      1056
           macro avg       0.76      0.73      0.72      1056
        weighted avg       0.76      0.73      0.72      1056
  
- **ResNet152V2:**
  - Accuracy: 81%
  - Precision, Recall, F1-Score for each class:
                       precision    recall  f1-score   support

            cataract       0.93      0.89      0.91       260
diabetic_retinopathy       0.84      0.86      0.85       275
            glaucoma       0.83      0.69      0.75       252
              normal       0.69      0.81      0.75       269

            accuracy                           0.81      1056
           macro avg       0.82      0.81      0.81      1056
        weighted avg       0.82      0.81      0.82      1056
  
- **EfficientNetB7:**
  - Accuracy: 25%
  - Precision, Recall, F1-Score for each class: 
 precision    recall  f1-score   support

            cataract       0.25      1.00      0.40       260
diabetic_retinopathy       0.00      0.00      0.00       275
            glaucoma       0.00      0.00      0.00       252
              normal       0.00      0.00      0.00       269

            accuracy                           0.25      1056
           macro avg       0.06      0.25      0.10      1056
        weighted avg       0.06      0.25      0.10      1056


## Visualization

- The training history (accuracy and loss) for each model is plotted to show the model's performance over epochs.
![image](https://github.com/user-attachments/assets/245cf367-8897-4aa3-9c3f-7579602b1b1b)
![image](https://github.com/user-attachments/assets/c04fea5d-8221-4c16-94c3-51c9e3793482)
![image](https://github.com/user-attachments/assets/6dfc3d9e-8d8c-4f30-b33b-3d98ca675ea7)


- Precision-Recall curves are plotted to evaluate the performance of each model across different thresholds.
- ![image](https://github.com/user-attachments/assets/42d30f79-e71a-496b-b35a-eeb27a11b9f5)


## Evaluation

The evaluation of each model includes:
- Confusion Matrix
- Classification Report
- Precision-Recall Curve

These metrics help in understanding the model's performance and identifying areas for improvement.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

