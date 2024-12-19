import os
import requests
from io import BytesIO

from PIL import Image
import streamlit as st
from st_clickable_images import clickable_images

from .model import classify_image, load_model, HF_MODELS
from .utils import image_to_base64, resize_image, select_image, convert_to_jpeg
from .dataset import get_image_path, load_datasets, show_dataset_details
from .paths import STATIC_PATH_CSS, STATIC_PATH_IMAGE, STATIC_PATH_JSON
from .guide import guidelines_classification
####################################################################
##          Clear specific keys in session state                  ##
####################################################################
def clear_session_state():
    """Clear relevant session state for a fresh inference cycle."""
    keys_to_clear = ['image', 'image_confirmed', 'show_results', 'predictions']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

class ImageClassification:
    def __init__(self):
        # Initialize available models
        self.models = HF_MODELS
        self.tree_datasets = load_datasets()
        self.placeholder_image = Image.open(os.path.join(STATIC_PATH_IMAGE, 'preview_placeholder.jpg')).convert('RGB')

        # Initialize session state
        if 'current_model' not in st.session_state:
            st.session_state.current_model = list(self.models.keys())[0]
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 0.5
        if 'top_k' not in st.session_state:
            st.session_state.top_k = 5
        if 'tree_datasets' not in st.session_state:
            st.session_state.tree_datasets = self.tree_datasets

    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded file, camera input, or image from advanced camera method"""
        if uploaded_file is not None:
            # Check if it's a file-like object (from st.camera_input or st.file_uploader)
            if hasattr(uploaded_file, 'read'):
                # Open image directly from file-like object
                image = convert_to_jpeg(Image.open(uploaded_file).convert('RGB'))
            # Check if it's already a PIL Image (from advanced camera method)
            elif isinstance(uploaded_file, Image.Image):
                image = uploaded_file
            else:
                st.error("Unsupported image format")
                return None

            image = resize_image(image, (224, 224))
            return image
        return None

    def about_tab(self):
        """About tab content"""
        st.title("Barkgods")

    def inference_tab(self):
        """Inference tab content for image classification"""
        col1, col2, col3 = st.columns([2, 1, 2])

        with col3:
            # Placeholder to toggle Advanced Settings visibility
            with st.container(border=True):
                advanced_settings = st.toggle("Advanced Settings", key='toggle_settings', value=True)
        # Show Advanced Settings in the main page if toggled
        if advanced_settings:
            with col3.expander("⚙️ Model Configuration", expanded=True):
                adv_set1, adv_set2 = st.columns(2)
                with adv_set1:
                    # Model selection
                    selected_model = st.selectbox(
                        "Select Model", 
                        list(self.models.keys())
                    )
                with adv_set2:
                    # Top-K predictions
                    top_k = st.selectbox(
                        "Number of Top Predictions",
                        [3, 5, 10]
                    )

                # Confidence threshold
                confidence_threshold = st.slider(
                    "Confidence Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.9, 
                    step=0.01
                )
                
                # Update session state
                st.session_state.current_model = selected_model
                st.session_state.confidence_threshold = confidence_threshold
                st.session_state.top_k = top_k

        
        if "show_results" not in st.session_state:
            st.session_state.show_results = False

        # Upload Method Selection
        with col1.container(border=True):
            st.markdown("<p><b><small>&#8595; Choose Upload Method</small></b></p>", unsafe_allow_html=True)
            upload_method = st.selectbox(
                "Choose Upload Method", 
                ["Upload Image", "Camera Capture", "Paste Image URL", "Example Images"],
                label_visibility='collapsed'
            )    
        if not st.session_state.show_results:
            # Create a full-width container for input methods
            with col1.container():
                if upload_method == "Upload Image":
                    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
                    st.session_state.image = self._process_uploaded_file(uploaded_file)
                elif upload_method == "Camera Capture":
                    enable = st.checkbox("Enabled Camera")
                    camera_image = st.camera_input("Capture Image", key='webcam', disabled= not enable, help='Allow permission to use the webcam')
                    st.session_state.image = self._process_uploaded_file(camera_image)
                elif upload_method == "Paste Image URL":
                    image_url = st.text_input("Paste Image URL")
                    if st.button("Submit URL", key="submit_url"):
                        if image_url:
                            try:
                                response = requests.get(image_url, timeout=10)  # Add a timeout
                                if response.status_code == 200:
                                    image = Image.open(BytesIO(response.content)).convert("RGB")  # Ensure RGB format
                                    image = convert_to_jpeg(image)
                                    image = resize_image(image, (224, 224))  # Resize image
                                    st.session_state.image = image  # Store in session state
                                else:
                                    st.error("Failed to retrieve image. Please check the URL.")
                            except Exception as e:
                                st.error(f"Error loading image: Invalid Url")
                                st.session_state.image = None
                        else:
                            st.warning("Please enter a valid URL.")

                elif upload_method == "Example Images":
                    st.session_state.image, _ = select_image(os.path.join(STATIC_PATH_JSON, 'tree_demos.json'))

                if "show_error_dialog" not in st.session_state:
                    st.session_state.show_error_dialog = False  # Track error dialog state

                # Image Classification Process
                if 'image' in st.session_state and st.session_state.image is not None:
                    image = st.session_state.image
                    selected_model = st.session_state.current_model
                    confidence_threshold = st.session_state.confidence_threshold
                    top_k = st.session_state.top_k
                    with col2.container(border=False):
                        st.image(image, caption='Uploaded/Captured Image', use_container_width=True, output_format='JPEG')
                        # Replace button with progress bar
                        if "classify_enabled" not in st.session_state:
                            st.session_state.classify_enabled = False  # Track button state

                        if not st.session_state.classify_enabled:
                            if st.button("Classify", use_container_width=True, type='primary'):
                                st.session_state.classify_enabled = True  # Button clicked
                                st.rerun()
                        else:
                            with st.spinner('Loading model...'):
                                model = load_model(selected_model)
                            with st.spinner('Scanning tree bark...'):
                                if model:
                                    predictions = classify_image(
                                        image,
                                        model,
                                        confidence_threshold,
                                        top_k
                                    )
                                    if predictions and len(predictions) > 0:
                                        st.session_state.predictions = predictions
                                        st.session_state.show_results = True
                                        # Reset session state to show the button again after completion
                                        st.session_state.classify_enabled = False
                                        # Redirect to results page
                                        st.switch_page('pages/result.py')
                                    else:
                                        st.session_state.show_error_dialog = True  # Trigger error dialog
                                        st.session_state.classify_enabled = False  # Reset classify state
                                        st.rerun()  # Trigger rerun for error dialog handling

                    # Show error dialog if triggered
                    if st.session_state.show_error_dialog:
                        @st.dialog("⚠️ No predictions found")
                        def input_error_dialog():
                            st.warning("Try the following:")
                            st.markdown("""
                            - **Check Image Quality**: Ensure clarity and focus on tree bark.
                            - **Adjust Confidence Threshold**: Lower the threshold for broader results.
                            - **Try Again**: Upload another image or use different model.
                            """)
                        input_error_dialog()
                        st.session_state.show_error_dialog = False  # Reset dialog state to prevent loop
                else:
                    with col2.container(border=True):
                        st.image(self.placeholder_image, caption='Upload an Image', output_format='JPEG', use_container_width=True)

        st.divider()
        # Image Upload Notice
        st.markdown("""
        <style>
        .privacy-notice {
            font-weight: 300; /* Light font weight */
            font-size: 14px; /* Optional: Adjust font size */
            color: #555; /* Optional: A subtle gray tone */
        }
        </style>

        <footer class="privacy-notice">
        🔒 Privacy Notice: Your image will be processed to generate model features. We won't store the image or the features on our server.
        </footer>
        """, unsafe_allow_html=True)
        # Instruction Button
        with st.expander("How to Use", icon="📘"):
            guidelines_classification()
    
    def datasets_tab(self):
        """Main function to display datasets in a grid layout."""
        st.title("Model Datasets")

        # Load cached datasets
        dataset_template = self.tree_datasets
        
        # Prepare the Base64 encoded images and titles
        image_paths = [f"data:image/jpeg;base64, {image_to_base64(get_image_path(dataset['image']['path']))}" for dataset in dataset_template]
        titles = [dataset['title'] for dataset in dataset_template]
        
        # Display clickable images
        clicked = clickable_images(
            image_paths,
            titles=titles,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px", "cursor": "pointer"},
            key="datasets_clickable_images"
        )
    
        # Show dialog for the clicked image
        if 'last_clicked' not in st.session_state:
            st.session_state.last_clicked = -1
            # st.toast(r"⚠️ Initialized state 'last_clicked'")

        if clicked != st.session_state.last_clicked and clicked > -1:
            st.session_state.last_clicked = clicked
            # st.toast(f"⚠️ Clicked:  {clicked}")
            selected_dataset = dataset_template[clicked]
            show_dataset_details(selected_dataset)

    def run(self):
        """Main app runner with sidebar control."""
        with open(os.path.join(STATIC_PATH_CSS, "style.css")) as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        listTabs = ["Inference", "Datasets", "About"]
        # Tab logic and session state tracking
        for i, tab in enumerate(st.tabs(listTabs)):
            with tab:
                st.session_state["active_tab"] = listTabs[i]
                if listTabs[i] == "Inference":
                    self.inference_tab()
                elif listTabs[i] == "Datasets":
                    self.datasets_tab()
                elif listTabs[i] == "About":
                    self.about_tab()
