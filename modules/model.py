from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch
import streamlit as st

HF_MODELS = {
    "ResNet-50": "alyzbane/2025-02-05-21-58-41-resnet-50",
    "Swin B/P4-W7": "alyzbane/2025-02-05-15-01-55-swin-base-patch4-window7-224",
    "ViT B/P16": "alyzbane/2025-02-05-14-22-36-vit-base-patch16-224",
    "ConvNeXT": "alyzbane/2025-02-10-08-48-20-convnextv2-tiny-1k-224",
}
# Caching the model loading process
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    """Load and return the selected model with its feature extractor and pipeline."""
    try:
        model_id = HF_MODELS.get(model_name)
        if not model_id:
            raise ValueError(f"Model '{model_name}' not found.")

        feature_extractor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)

        # Create and return the classification pipeline
        classifier = pipeline(
            "image-classification", 
            model=model, 
            feature_extractor=feature_extractor,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        return classifier
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def classify_image(image, model, confidence_threshold, top_k):
    """Classify image with confidence filtering and name mapping."""
    try:
        # Perform classification
        predictions = model(image, top_k=top_k)
        name_mapping = st.session_state.common_to_scientific
        # Filter predictions based on confidence threshold and add common names
        filtered_predictions = []
        for pred in predictions:
            if pred['score'] >= confidence_threshold:
                common_name = pred['label']
                # Get the common name from the mapping
                scientific_name = name_mapping.get(common_name, "Unknown")
                
                new_pred = {
                    "score": pred["score"],
                    "scientific_name": scientific_name,
                    "common_name": common_name
                }
                
                filtered_predictions.append(new_pred)
        return filtered_predictions
    except Exception as e:
        st.error(f"Classification error: {e}")
        return []