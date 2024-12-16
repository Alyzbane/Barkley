import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline
import streamlit as st

# Caching the model loading process
@st.cache_resource
def load_model(model_name):
    """Load and return the selected model with its feature extractor and pipeline."""
    models = {
        "ResNet-50": "models/ResNet-50",
        "ViT Base": "models/ViT-base-patch16",
        "ConvNeXT": "models/ConvNeXT",
        "Swin Base": "models/Swin-base-patch4-window7",
    }

    try:
        model_id = models.get(model_name)
        if not model_id:
            raise ValueError(f"Model '{model_name}' not found.")

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create and return the classification pipeline
        classifier = pipeline(
            "image-classification", 
            model=model, 
            feature_extractor=feature_extractor,
            device=device,
        )
        return classifier
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
