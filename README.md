# PG05
AY2425-S2 ST5188 PG05: Three categories of CXR
This project implements a CXR image classification task based on PyTorch and incorporates Grad-CAM technology for model visualization. The project supports data preprocessing, model training, validation, testing, and result visualization.

The file is named after the pretrained models used, with ResNet18 and EfficientNet being the primary models we experimented with. It includes the following features:

**Data Preprocessing**: Supports operations such as image resizing, random horizontal flipping, and random rotation.

**Model Definition**: Custom models based on ResNet18/50/101 and EfficientNet, with support for feature extraction module extensions such as Gabor Filters.

**Training and Validation**: Uses K-fold cross-validation for model training and supports dynamic saving of model parameters.

**Testing and Evaluation**: Calculates test set accuracy and generates a confusion matrix.

**Grad-CAM Visualization**: Generates Grad-CAM heatmaps for misclassified samples to help understand model decisions.

## About Dataset:

The dataset structure is described in the "Data Preprocessing" code block in the file.

The original dataset comes from https://www.kaggle.com/datasets/unaissait/curated-chest-xray-image-dataset-for-covid19?utm_source=chatgpt.com

For the dataset we used for training, three preprocessing steps were applied: merging the COVID-19 and Pneumonia-Viral classes, removing duplicates, and cropping the lung regions. The corresponding processed dataset is available at https://drive.google.com/drive/folders/1iEyI_RQc_DbJZmpQQxlX3ewfO_EUtBrh?usp=sharing

We also provide processing code in the "find_similar" folder to identify identical images, as well as text files containing the names of identical images under each category.

## About Environment and GPU Hardware Configuration:
The environment is specified in the requirements.txt, and the hardware configuration for training the model is as follows:

Graphics Card: RTX 4060 Laptop GPU, CUDA Version: 12.6

Memory: 8GB GDDR6

Equipped with 3072 CUDA cores and 4th Gen Tensor Cores

**Note:** The versions provided in requirements.txt are compatible with CUDA 12.6. If your CUDA version is different, please select versions of certain packages, such as PyTorch and torchvision, that are compatible with your CUDA version.

## About Preprocessing:
The pre-trained model used for lung cropping is from: https://github.com/IlliaOvcharenko/lung-segmentation
The pre-trained model used for denoising is from: https://github.com/swz30/MPRNet
