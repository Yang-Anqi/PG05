# PG05
AY2425-S2 ST5188 PG05: Three categories of CXR
This project implements a CXR image classification task based on PyTorch and incorporates Grad-CAM technology for model visualization. The project supports data preprocessing, model training, validation, testing, and result visualization.

The file is named after the pretrained models used, with ResNet18 and EfficientNet being the primary models we experimented with. It includes the following features:

Data Preprocessing: Supports operations such as image resizing, random horizontal flipping, and random rotation.

Model Definition: Custom models based on ResNet18/50/101 and EfficientNet, with support for feature extraction module extensions such as Gabor Filters.

Training and Validation: Uses K-fold cross-validation for model training and supports dynamic saving of model parameters.

Testing and Evaluation: Calculates test set accuracy and generates a confusion matrix.

Grad-CAM Visualization: Generates Grad-CAM heatmaps for misclassified samples to help understand model decisions.
