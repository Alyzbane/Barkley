import os
from .redis import get_redis_connection
from .utils import hash_image
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline,
    AutoProcessor,
    AutoModelForCausalLM,
)
import streamlit as st

# Constants
HF_MODELS = {
    "ResNet-50": "models/resnet-50",
    "Swin B/P4-W7": "models/swin",
    "ViT B/P16": "models/vit",
    "ConvNeXT": "models/convnext",
}
DEVICE = "cpu" # streamlit community cloud is running on CPU
DTYPE = torch.float32
redis_conn = get_redis_connection()

# ======================================
# Deteector Model Loading and Inference 
# ======================================
# workaround for unnecessary flash_attn requirement
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

# Set the Florence-2 model as detector model
@st.cache_resource(show_spinner=False)
def load_detector_model():
    florence2_model = "microsoft/florence-2-base"
    try:
        model_id = florence2_model
        if not model_id:
            raise ValueError(f"Model '{model_id}' not found.")

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): # workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa", torch_dtype=DTYPE,trust_remote_code=True)

        return processor, model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# Run detection model via caption generation
def run_florence_inference(task_prompt, text_input=None):
    image = st.session_state.image
    processor, model = load_detector_model()
    
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    return generated_text, image.size  # Return raw text and image size for later parsing

# @st.cache_data(ttl=600, max_entries=50, show_spinner=False)
def parse_florence_output(generated_text, task_prompt, image_size):
    processor, _ = load_detector_model()
    
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=image_size
    )
    return parsed_answer

def run_florence(task_prompt, text_input=None):
    image_hash = hash_image(st.session_state.image)
    prompt = task_prompt if text_input is None else task_prompt + text_input
    cache_key = f"florence:{image_hash}:{prompt}"

    cached = redis_conn.get_json(cache_key)
    if cached:
        return cached

    generated_text, image_size = run_florence_inference(task_prompt, text_input)
    parsed = parse_florence_output(generated_text, task_prompt, image_size)
    redis_conn.set_json(cache_key, parsed, ttl=600)
    return parsed

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

        feature_extractor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)

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
            redis_conn.set_json(cache_key, predictions, ttl=600)
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




