from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch
import streamlit as st

HF_MODELS = {
    "Swin B/P4-W7": "alyzbane/swin-base-patch4-window7-224-finetuned-barkley",
    "ViT B/P16": "alyzbane/vit-base-patch16-224-finetuned-barkley",
    "ResNet-50": "alyzbane/resnet-50-finetuned-FBark-5",
    "ConvNeXT": "alyzbane/convnext-tiny-224-finetuned-barkley",
}
# Caching the model loading process
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    """Load and return the selected model with its feature extractor and pipeline."""
    # models = { # Localhost testing
    #     "ResNet-50": "models/ResNet-50",
    #     "ViT Base": "models/ViT-base-patch16",
    #     "ConvNeXT": "models/ConvNeXT",
    #     "Swin Base": "models/Swin-base-patch4-window7",
    # }

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

        # Define the mapping of scientific names to common names
        name_mapping = {
            "Roystonea regia": "Royal Palm",
            "Pterocarpus indicus": "Narra",
            "Tabebuia": "Trumpet",
            "Mangifera indica": "Mango",
            "Iinstia bijuga": "Ipil"
        }

        # Filter predictions based on confidence threshold and add common names
        filtered_predictions = []
        for pred in predictions:
            if pred['score'] >= confidence_threshold:
                # Update scientific name if it matches "Iinstia bijuga"
                scientific_name = pred['label']
                if scientific_name == "Iinstia bijuga":
                    scientific_name = "Cananga odorata"
                
                # Get the common name from the mapping
                common_name = name_mapping.get(pred['label'], "Unknown")
                
                # Create a new dictionary for the prediction
                new_pred = {
                    "score": pred["score"],
                    "label": scientific_name,  # Original label
                    "scientific_name": scientific_name,  # Updated scientific name
                    "common_name": common_name  # Common name
                }
                
                filtered_predictions.append(new_pred)
        return filtered_predictions
    except Exception as e:
        st.error(f"Classification error: {e}")
        return []
