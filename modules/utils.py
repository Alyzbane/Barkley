import json
import os
import io
import base64

from PIL import Image
from streamlit_image_select import image_select
from .paths import STATIC_PATH_IMAGE

#################################################
##             Json loader
#################################################
def load_json(json_path):
    """Load the tree species data from a JSON file."""
    with open(json_path, 'r') as file:
        return json.load(file)

#################################################
##              Resizing Image
#################################################
def resize_image(image, size):
    """Resize image while maintaining aspect ratio"""
    image.resize(size, Image.LANCZOS)
    return image
#################################################
##             Image conversion
#################################################
def convert_to_jpeg(image):
    # Convert image to JPEG format
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    jpeg_image = Image.open(buffer)  # Reload the image as a PIL object
    return jpeg_image

#################################################
##        Interactive Image Selector
#################################################
def select_image(path):
    """
        Image selection.

        path - Path containing images
    """

    import streamlit as st
    
    # Load the JSON data
    tree_data = load_json(path)

    # Prepare the image paths and keys (names) for the selection
    image_paths = [os.path.join(STATIC_PATH_IMAGE, 'examples', tree["image_path"]) for tree in tree_data.values()]
    names = list(tree_data.keys())  # Use the keys (e.g., "Royal Palm") as captions

    # Image selection
    selected_image_path = image_select(
        label="Select an Image for Classification",
        images=image_paths,
        captions=names,  # Use keys (names) as captions
        use_container_width=True,
    )

    if selected_image_path:
        # Find the selected tree name based on the image path
        selected_name = None
        for name, tree in tree_data.items():
            if tree["image_path"] == selected_image_path:
                selected_name = name
                break

        # Load the selected image
        selected_image = Image.open(selected_image_path).convert('RGB')
        return selected_image, selected_name  # Return the image and name
    return None, None



#################################################
##        Image conversion to raw format
#################################################
def image_to_base64(image_path):
    """Convert an image file to base64 format."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
