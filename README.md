# Barkley : Tree Species Classification with Streamlit

This project aims to develop a practical and accurate deep learning-based system for identifying tree species from images of tree bark. The system leverages various deep learning architectures, including CNNs and Vision Transformers, to classify tree species based on bark images.

## **Table of Contents**
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [References](#references)

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Alyzbane/Barkley.git
   cd Barkley
2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt

## **Usage**
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
2. **Open your browser and go to:**
   ```bash
   http://localhost:8501
3. **Upload an image of tree bark and select a model to predict the tree species.**

## **Models**

This project leverages several state-of-the-art deep learning architectures for tree species classification based on tree bark images. Below are the models used in this system:

| **Model Name**            | **Link**                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| **ResNet-50**              | [ResNet-50 on Hugging Face](https://huggingface.co/alyzbane/resnet-50-finetuned-barkley)  |
| **Vision Transformer (ViT)** | [Vision Transformer (ViT) on Hugging Face](https://huggingface.co/google/vit-large-patch16-224) |
| **Swin Transformer**       | [Swin Transformer on Hugging Face](https://huggingface.co/alyzbane/swin-base-patch4-window7-224-finetuned-barkley) |
| **ConvNeXt**               | [ConvNeXt on Hugging Face](https://huggingface.co/alyzbane/convnext-tiny-224-finetuned-barkley) |


Each of these models has been trained to identify tree species based on the visual features of tree bark images. The performance of each model is compared in this project to determine the best approach for tree species classification.


## **References**

The following resources were referenced during the development of this project:

1. **ResNet-50**  
   - [Hugging Face - ResNet](https://huggingface.co/docs/transformers/en/model_doc/resnet)  
   A convolutional neural network (CNN) designed for image classification, with deep residual connections for improved accuracy.

2. **Vision Transformer (ViT)**  
   - [Hugging Face - Vision Transformer](https://huggingface.co/docs/transformers/en/model_doc/vit)  
   A transformer-based model that applies self-attention mechanisms to image patches for effective image classification.

3. **Swin Transformer**  
   - [Hugging Face - Swin Transformer](https://huggingface.co/docs/transformers/en/model_doc/swin)  
   A vision transformer model optimized for hierarchical representation of images across multiple scales.

4. **ConvNeXt**  
   - [Hugging Face - ConvNeXt](https://huggingface.co/docs/transformers/en/model_doc/convnext)  
   A modernized convolutional neural network architecture, optimized for performance and efficiency in image classification tasks.
