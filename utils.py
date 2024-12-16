import json
import os

from PIL import Image
from streamlit_image_select import image_select

STATIC_IMAGE_PATH = 'static/images/'
STATIC_IMAGE_JSON = 'static/json/'

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
##        Interactive Image Selector
#################################################
def select_image(path):
    """
        Image selection.

        path - Path containing images
    """
    # Load the JSON data
    tree_data = load_json(path)

    # Prepare the image paths and keys (names) for the selection
    image_paths = [os.path.join(STATIC_IMAGE_PATH, 'examples', tree["image_path"]) for tree in tree_data.values()]
    names = list(tree_data.keys())  # Use the keys (e.g., "Royal Palm") as captions

    # Image selection
    selected_image_path = image_select(
        label="Select an Image for Classification",
        images=image_paths,
        captions=names,  # Use keys (names) as captions
        use_container_width=True
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
