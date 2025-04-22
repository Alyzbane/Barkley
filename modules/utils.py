import json
import os
import io
import base64
import requests
import hashlib
import streamlit as st

from PIL import Image
from streamlit_image_select import image_select
from .paths import STATIC_PATH_IMAGE, STATIC_PATH_CSS

# #################################################
# ##             SVG files loader
# #################################################
# def load_static_svg(filename: str):
#     with open(os.path.join(STATIC_PATH_SVG, filename), 'r') as file:
#         svg_content = file.read()
#     return svg_content

#################################################
##             Json loader
#################################################
def load_json(url=None, path=None):
    """Load the tree species data from a URL or sJSON file."""
    if url is not None:
        requests.get(url)
        return requests.get(url).json()
    else:
        with open(path, 'r') as file:
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
    image.save(buffer, format="JPEG", quality=80)
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
    # Load the JSON data
    tree_data = load_json(path=path)

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
        selected_image = resize_image(selected_image, (224, 224))  # Resize to 224x224
        return selected_image, selected_name  # Return the image and name
    return None, None

#################################################
##        Image conversion to raw format
#################################################
@st.cache_data(ttl=60*60)
def image_to_base64(image_path: str):
    """Convert an image file to base64 format."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

#################################################
##        Return the css code in string format
#################################################
def get_css(filename: str) -> str:
    with open(os.path.join(STATIC_PATH_CSS, filename)) as f:
        css_code = f.read()
    return f"<style>{css_code}</style>"

#################################################
##        Getting the css and loading
#################################################
@st.cache_resource(ttl=3600)  # Cache expires after 1 hour
def load_css(filename: str):
    css_code = get_css(filename)
    st.markdown(css_code, unsafe_allow_html=True)


######################################################
##   Processing jpeg, jpg, png, etc. image files    ##
######################################################
def __process_uploaded_file(uploaded_file):
    """Process uploaded file, camera input, or image from advanced camera method"""
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = resize_image(image, (224, 224))  # Resize to 224x224
        return image
    return None

###################################################
##        Detect tree or trunk in caption        ##
###################################################
def detect_caption(caption: str) -> bool:
    keywords = ['bark',
                'tree bark',
                'tree trunk',
                'trunk',
                'tree',]
    caption_lower = caption.lower()
    return any(keyword in caption_lower for keyword in keywords)

###################################################
##       Hash image into MD% for cache           ##
###################################################
def hash_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return hashlib.md5(img_bytes).hexdigest()