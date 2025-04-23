import os
from .redis import get_redis_connection
from .utils import hash_image
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline,
    AutoProcessor,
    AutoModelForCausalLM,
)
import streamlit as st

# Constants
HF_MODELS = {
    "ResNet-50": "alyzbane/2025-02-05-21-58-41-resnet-50",
    "Swin B/P4-W7": "alyzbane/2025-02-05-15-01-55-swin-base-patch4-window7-224",
    "ViT B/P16": "alyzbane/2025-02-05-14-22-36-vit-base-patch16-224",
    "ConvNeXT": "alyzbane/2025-02-10-08-48-20-convnextv2-tiny-1k-224",
}
DEVICE = "cpu" # streamlit community cloud is running on CPU
DTYPE = torch.float32
REDIS_TTL = 12*60*60
redis_conn = get_redis_connection()

# ======================================
# Deteector Model Loading and Inference 
# ======================================
@st.cache_resource(show_spinner=False)
def load_captioning_pipeline():
    """Load the ViT-GPT2 image captioning pipeline."""
    try:
        return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=DEVICE)
    except Exception as e:
        raise RuntimeError(f"Error loading image captioning pipeline: {e}")
    
def run_captioning_inference(image_path):
    """Run the ViT-GPT2 pipeline for image captioning."""
    captioning_pipeline = load_captioning_pipeline()

    # Generate caption
    result = captioning_pipeline(image_path)
    if result and "generated_text" in result[0]:
        return result[0]["generated_text"].strip()
    else:
        raise ValueError("Failed to generate caption.")
    
def run_captioning():
    """Run the captioning pipeline with caching."""
    image = st.session_state.image
    image_hash = hash_image(image)
    cache_key = f"captioning:{image_hash}"

    # Check cache
    cached_caption = redis_conn.get_json(cache_key)
    if cached_caption:
        return cached_caption

    # Generate caption and cache it
    caption = run_captioning_inference(image)
    redis_conn.set_json(cache_key, caption, ttl=REDIS_TTL)
    return caption

# ================================
# Model Loading and Classification
# ================================
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    """Load and return the selected model with its feature extractor and pipeline."""
    try:
        model_id = HF_MODELS.get(model_name)
        if not model_id:
            raise ValueError(f"Model '{model_name}' not found.")

        feature_extractor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True)

        # Create and return the classification pipeline
        classifier = pipeline(
            "image-classification", 
            model=model, 
            feature_extractor=feature_extractor,
            device = DEVICE,
        )
        return classifier
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def classify_image_inference(model_name, top_k):
    model = load_model(model_name) 
    image = st.session_state.image
    predictions = model(image, top_k=top_k)  # Run raw predictions (uncached)
    return predictions

# @st.cache_data(ttl=600, max_entries=50, show_spinner=False)
def filter_predictions(predictions, confidence_threshold, name_mapping, top_k):
    """Filter predictions and add scientific names."""
    filtered_predictions = []
    confidence_threshold = confidence_threshold / 100.0
    for pred in predictions[:top_k]:
        if pred['score'] >= confidence_threshold:
            common_name = pred['label']
            scientific_name = name_mapping.get(common_name, "Unknown")

            filtered_predictions.append({
                "score": pred["score"],
                "scientific_name": scientific_name,
                "common_name": common_name,
            })
    
    return filtered_predictions

def classify_image(model_name):
    try:
        image_hash = hash_image(st.session_state.image)
        cache_key = f"classify_raw:{image_hash}:{model_name}"
        
        cached_predictions = redis_conn.get_json(cache_key)
        if not cached_predictions:
            predictions = classify_image_inference(model_name, top_k=13)  # max K
            redis_conn.set_json(cache_key, predictions, ttl=REDIS_TTL)
        else:
            predictions = cached_predictions

        # Now apply filtering on the cached data
        filtered_predictions = filter_predictions(
            predictions,
            confidence_threshold=st.session_state.confidence_threshold,
            name_mapping=st.session_state.common_to_scientific,
            top_k=st.session_state.top_k
        )

        return filtered_predictions

    except Exception as e:
        return {"error": str(e)}




