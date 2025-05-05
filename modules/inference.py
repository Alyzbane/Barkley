from PIL import Image
import os
import requests
from io import BytesIO

import streamlit as st

from .model import classify_image, HF_MODELS, run_captioning
from .utils import resize_image, select_image, __process_uploaded_file, detect_caption
from .paths import STATIC_PATH_IMAGE, STATIC_PATH_JSON
from views.guide import display_guidelines 

NUM_CLASSES = 13
####################################################################
##          Clear specific keys in session state                  ##
####################################################################
def clear_session_state():
    """Clear relevant session state for a fresh inference cycle."""
    keys_to_clear = ['image', 'image_confirmed', 'show_results', 'predictions', 'top_k', 'confidence_threshold', 'current_model']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def instructions_expander():
    """Display instructions with an expander."""
    with st.expander(":scroll: Guidelines", expanded=False):
        display_guidelines()

@st.fragment()
def handle_advanced_settings():
    """Handle advanced settings UI and logic"""
    st.write("⚙️ Model Configuration")
    selected_model = st.selectbox(
        "Select Model", 
        list(HF_MODELS.keys()),
        help="Select the model you want to use for predictions."
    )
    top_k = st.slider(
        "Top Predictions",
        1, NUM_CLASSES, 5,
        help="Choose the number of top predictions to display."
    )

    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0, 
        max_value=100, 
        value=0,
        step=1,
        help="Higher values mean the model will return more confident predictions."
    )

    st.session_state.current_model = selected_model
    st.session_state.confidence_threshold = confidence_threshold
    st.session_state.top_k = top_k

def handle_image_input(col1, col2, placeholder_image):
    """Handle image input methods and processing"""
    with col1.container(border=True):
        st.markdown("<p><b><small>&#8595; Choose Upload Method</small></b></p>", unsafe_allow_html=True)
        upload_method = st.selectbox(
            "Choose Upload Method", 
            ["Upload Image", "Camera Capture", "Paste Image URL", "Example Images"],
            label_visibility='collapsed'
        )

    if not st.session_state.get('show_results', False):
        with col1.container():
            if upload_method == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
                st.session_state.image = __process_uploaded_file(uploaded_file)
            
            elif upload_method == "Camera Capture":
                enable = st.checkbox("Enabled Camera")
                camera_image = st.camera_input("Capture Image", key='webcam', disabled=not enable)
                st.session_state.image = __process_uploaded_file(camera_image)
            
            elif upload_method == "Paste Image URL":
                handle_url_input()
            
            elif upload_method == "Example Images":
                st.session_state.image, _ = select_image(os.path.join(STATIC_PATH_JSON, 'tree_demos.json'))

            display_image_and_classify(col2, placeholder_image)

def handle_url_input():
    """Handle URL input method"""
    image_url = st.text_input("Paste Image URL")
    if st.button("Submit URL", key="submit_url"):
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image = resize_image(image, (224, 224))  # Resize to 224x224
                    st.session_state.image = image
                else:
                    st.error("Failed to retrieve image. Please check the URL.")
            except Exception as e:
                st.error("Error loading image: Invalid Url")
                st.session_state.image = None
        else:
            st.warning("Please enter a valid URL.")

def display_image_and_classify(col2, placeholder_image):
    """Display uploaded image and handle classification"""
    if 'image' in st.session_state and st.session_state.image is not None:
        image = st.session_state.image
        with col2.container(border=True):
            st.image(image, caption='Uploaded/Captured Image', use_container_width=True)
            if "classify_enabled" not in st.session_state:
                st.session_state.classify_enabled = False
            if not st.session_state.classify_enabled:
                if st.button("Classify", use_container_width=True, type='primary'):
                    st.session_state.classify_enabled = True
                    st.rerun()
            else:
                handle_classification()
    else:
        with col2.container(border=True):
            st.image(placeholder_image, caption='Upload an Image', output_format='JPEG', use_container_width=True)

@st.fragment()
def handle_detection():
    # Create a progress bar for the entire process
    progress_bar = st.progress(0, text="Starting analysis...")
    
    # Detection phase (25% of the process)
    progress_bar.progress(10, text="Detecting tree bark...")
    caption_texts = run_captioning()
    progress_bar.progress(30, text="Detecting tree bark...")
    if not detect_caption(caption_texts):
        progress_bar.progress(100, text="Detection failed")
        st.session_state.show_error_dialog = True
        st.session_state.classify_enabled = False
        st.rerun()
    progress_bar.progress(60, text="Detecting tree bark...")
    return progress_bar

def handle_classification():
    """Handle the classification process with progress bars"""
    progress_bar = handle_detection()
    progress_bar.progress(70, text="Analyzing tree bark...")
    predictions = classify_image(st.session_state.current_model)
    
    progress_bar.progress(100, text="Analysis complete")
    if predictions and len(predictions) > 0:
        st.session_state.predictions = predictions
        st.session_state.show_results = True
        st.session_state.classify_enabled = False
        st.switch_page('pages/result.py')
    else:
        st.session_state.show_error_dialog = True
        st.session_state.classify_enabled = False
        st.rerun()

@st.fragment()
def display_error_dialog():
    """Display error dialog when no predictions are found"""
    if st.session_state.get('show_error_dialog', False):
        @st.dialog("⚠️ No predictions found")
        def input_error_dialog():
            st.warning("Try the following:")
            st.markdown("""
            - **Check Image Quality**: Ensure clarity and focus on tree bark.
            - **Adjust Confidence Threshold**: Lower the threshold for broader results.
            - **Explore Example Images**: Use the example images for testing.
            - **Try Again**: Upload another image or use different model. 
            """)
        input_error_dialog()
        st.session_state.show_error_dialog = False

def inference_tab():
    """Main inference tab function"""
    st.markdown("""
    <div style="margin-bottom: 1em;">
        <p style="text-align: center;">Explore the fascinating world of trees through their unique <i>bark</i> patterns. 
           Whether you're a botanist, researcher, or nature enthusiast, Barkley helps you 
           identify tree species with just a photo 📸.</p>
    </div>
    """, unsafe_allow_html=True)

    placeholder_image = Image.open(os.path.join(STATIC_PATH_IMAGE, 'preview_placeholder.jpg')).convert('RGB')
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with st.sidebar:
        handle_advanced_settings()

    st.markdown(
        """
        <style>

        </style>
        """,
        unsafe_allow_html=True
    )

    handle_image_input(col1, col2, placeholder_image)
    with col3:
        instructions_expander()
    display_error_dialog()
