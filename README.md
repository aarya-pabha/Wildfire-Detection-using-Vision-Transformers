# Wildfire Detection Using Vision Transformers
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Accuracy](https://img.shields.io/badge/Binary%20Accuracy-97.8%25-green)

This project explores the use of Vision Transformers (ViTs) for early wildfire detection through image classification. The cascading ViT model employed in this project improves the accuracy of fire and no-fire classification using multiple dedicated submodels.

## Project Overview

Wildfires pose significant risks to ecosystems and human life, and early detection is crucial for effective intervention. This project leverages the power of Vision Transformers (ViT) for wildfire image classification with the following goals:
- Binary classification of images into **Fire** and **No-Fire**.
- Subclassification of both fire and no-fire images into specific categories (e.g., smoke, fire, shadows).

## Architecture

The pipeline uses a two-stage cascading approach:
1. A binary ViT classifier routes each image to Fire or No-Fire
2. Dedicated subclass ViT models then isolate confounding visual elements 
   (smoke, fog, shadows) within each branch

This routing design was validated against 23 model variations including 
Swin, DeiT, and BEiT architectures.

### Key Features
- **Binary ViT Classifier**: Classifies images as fire or no-fire with an accuracy of 97.8%.
- **Cascading Submodels**: Dedicated ViT models further classify fire and no-fire subclasses, achieving an overall accuracy of 88.54%.

## Dataset

- **Original Dataset**: The dataset used for training is [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/versions/2). It consists of 2700 aerial and ground-based images classified into **fire** and **no-fire** categories, with subclasses such as smoke and confounding elements.
  
- **Validation Dataset (Updated)**: Due to a corrupted file in the original validation dataset, the updated version can be found here: [Validation Dataset (Edited)](https://www.kaggle.com/datasets/aaryamahajanpabha/validation-edited).

## Results

The model achieved the following metrics:
- **Binary ViT Accuracy**: 97.8%
- **Cascading Model Accuracy**: 88.54%
- **Comparison with Dual-Dataset DL**: Outperformed existing models in accuracy and precision.

## Results

| Model | Accuracy | Precision | ROC-AUC |
|---|---|---|---|
| Binary ViT (ours) | 97.8% | — | 0.975 |
| Cascading ViT (ours) | 88.54% | — | — |
| Dual-Dataset DL Baseline | 95.7% | lower | lower |
| CNN Baseline | ~95.7% | lower | lower |

## Final Models

The final pre-trained models used in this project can be found at the following locations:
- [Binary ViT Model (10 Epochs)](https://www.kaggle.com/datasets/aaryamahajanpabha/capstone-models?select=Binary_VIT_10epochs)
- [Fire Subclass Model](https://www.kaggle.com/datasets/aaryamahajanpabha/capstone-models?select=fire_subclass_model_VIT)
- [No-Fire Subclass Model (10 Epochs)](https://www.kaggle.com/datasets/aaryamahajanpabha/capstone-models?select=nofire_subclass_model_VIT_10epochs)

For more details, refer to the [models_used.md](models_used.md) file.

## How to Run the Project

1. Clone this repository:
    ```bash
    git clone https://github.com/aarya-pabha/Wildfire-Detection-using-Vision-Transformers.git
    ```
2. Install dependencies:
    ```bash
    pip install transformers torch matplotlib seaborn
    ```
3. Download the dataset from the links provided above.
4. Download the pre-trained models from Kaggle and place them in a folder named `models/`.
5. Run the demo code available in the repository to see the wildfire detection in action.

## Demo Code

The final demo code for running the wildfire detection model is available in the repository, including the cascading model used for classification. The code also includes steps for visualizing attention maps to explain model predictions.

## How to Train the Model

You can also find the code to train the submodels of the cascading model in the `training/` folder. These scripts use the original dataset to fine-tune Vision Transformer models for binary and subclass classification.

## Future Work
- **Hyperparameter tuning** for further performance improvement.
- **Transfer learning** to enhance model robustness on larger datasets.
- **Explainability** improvements through more detailed attention maps.

## Acknowledgments
This project is inspired by various research papers exploring the potential of Vision Transformers for image classification tasks.
