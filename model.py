from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import streamlit as st
HF_MODELS = {
    "ResNet-50": "alyzbane/resnet-50-finetuned-barkley",
    "ViT Base": "alyzbane/vit-base-patch16-224-finetuned-barkley",
    "ConvNeXT": "alyzbane/convnext-tiny-224-finetuned-barkley",
    "Swin Base": "alyzbane/swin-base-patch4-window7-224-finetuned-barkley",
}
# Caching the preloaded models
@st.cache_resource(show_spinner="Loading all models...")
def preload_all_models():
    """Preload all models and return a dictionary of pipelines."""
    # Dictionary to store pipelines
    model_pipelines = {}

    for model_name, model_id in HF_MODELS.items():
        try:
            # Load the feature extractor and model
            feature_extractor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)

            # Create the pipeline
            classifier = pipeline(
                "image-classification",
                model=model,
                feature_extractor=feature_extractor,
            )

            # Store the pipeline in the dictionary
            model_pipelines[model_name] = classifier
        except Exception as e:
            st.warning(f"Could not load model {model_name}: {e}")

    return model_pipelines

# Caching the model loading process
@st.cache_resource(show_spinner="Loading model...")
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
            "Iinstia bijuga": "Ilang-ilang"
        }

        # Filter predictions based on confidence threshold and add common names
        filtered_predictions = []
        for pred in predictions:
            if pred['score'] >= confidence_threshold:
                # Get the common name from the mapping
                common_name = name_mapping.get(pred['label'], "Unknown")
                
                # Create a new dictionary for the prediction
                new_pred = {
                    "score": pred["score"],
                    "label": pred["label"],  # Original label
                    "scientific_name": pred["label"],  # Same as original label
                    "common_name": common_name  # Common name
                }
                
                filtered_predictions.append(new_pred)

        return filtered_predictions
    except Exception as e:
        st.error(f"Classification error: {e}")
        return []
