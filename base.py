import os
import json
import time
import requests
from io import BytesIO

from PIL import Image
import streamlit as st

from model import preload_all_models, classify_image, load_model
from utils import resize_image, select_image
from dataset import load_datasets, show_dataset_details

class ImageClassification:
    def __init__(self):
        # Initialize available models
        # self.models = {
        #     "ResNet-50": "models/ResNet-50",
        #     "ViT Base": "models/ViT-base-patch16",
        #     "ConvNeXT": "models/ConvNeXT",
        #     "Swin Base": "models/Swin-base-patch4-window7",
        # }
        self.models = preload_all_models() # Bug: Automatically going back to the introduction tab  when inferencing for the first time
        self.tree_datasets = load_datasets()

        # Initialize session state
        if 'current_model' not in st.session_state:
            st.session_state.current_model = list(self.models.keys())[0]
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 0.5
        if 'top_k' not in st.session_state:
            st.session_state.top_k = 5

    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded file, camera input, or image from advanced camera method"""
        if uploaded_file is not None:
            # Check if it's a file-like object (from st.camera_input or st.file_uploader)
            if hasattr(uploaded_file, 'read'):
                # Open image directly from file-like object
                image = Image.open(uploaded_file)
            # Check if it's already a PIL Image (from advanced camera method)
            elif isinstance(uploaded_file, Image.Image):
                image = uploaded_file
            else:
                st.error("Unsupported image format")
                return None

            # Resize image
            image = resize_image(image, (224, 224))
            return image
        return None

    def introduction_tab(self):
        """Introduction tab content"""
        st.title("Image Classification App")
        st.markdown("""
        ### Welcome to the Advanced Image Classification Dashboard
        
        Explore state-of-the-art image classification using transformer models:
        
        - Multiple pre-trained models
        - Confidence threshold filtering
        - Interactive image upload
        - Detailed classification insights
        """)

    @st.dialog("Image Classification Guide")
    def show_help_dialog(self):
        """Display help dialog using st.dialog"""
        st.markdown("""
        ## Image Classification Guide
        ### Best Practices for Image Selection
        - Use clear, well-lit images
        - Ensure the main subject is centered
        - Avoid complex backgrounds

        ### Image Tips
        ✅ **Good images**:
        - Singular, clear object
        - Minimal background noise
        - Sharp focus

        ❌ **Avoid**:
        - Blurry images
        - Cluttered scenes
        - Small or distant objects
        """)

    def inference_tab(self):
        """Inference tab content for image classification"""
        # Help Button
        if st.button("📘 How to Use"):
            self.show_help_dialog()
            
        # Sidebar configuration (remains the same as previous implementation)
        with st.sidebar:
            st.header("Model Configuration")
            
            # Model selection
            selected_model = st.selectbox(
                "Select Model", 
                list(self.models.keys())
            )
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01
            )
            
            # Top-K predictions
            top_k = st.selectbox(
                "Number of Top Predictions",
                [3, 5, 10]
            )
            
            # Update session state
            st.session_state.current_model = selected_model
            st.session_state.confidence_threshold = confidence_threshold
            st.session_state.top_k = top_k

        # Upload Method Selection
        upload_method = st.radio(
            "Choose Upload Method", 
            ["Upload Image", "Camera Capture", "Paste Image URL", "Example Images"],
            horizontal=True
        )

        # Image Upload Notice
        st.markdown("""
        🔒 **Privacy Notice**: 
        Your image will be processed to generate model features. 
        We won't store the image or the features on our servers.
        """)
        
        image = None

        # Create a full-width container for input methods
        with st.container():
            if upload_method == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
                image = self._process_uploaded_file(uploaded_file)

            elif upload_method == "Camera Capture":
                enable = st.checkbox("Enabled Camera")
                camera_image = st.camera_input("Capture Image", key='webcam', disabled= not enable, help='Allow permission to use the webcam')
                image = self._process_uploaded_file(camera_image)
            
            elif upload_method == "Paste Image URL":
                image_url = st.text_input("Paste Image URL")
                if st.button("Submit URL"):
                    if image_url:
                        try:
                            response = requests.get(image_url)
                            image = Image.open(BytesIO(response.content))
                            image = resize_image(image, (224, 224))  # Resize to fit better in layout
                        except Exception as e:
                            st.error("Invalid URL or unable to download image")
                            image = None
            elif upload_method == "Example Images":
                image, _ = select_image(r'static/json/trees.json')

        # Image Classification Process
        if image:
            col1, col2 = st.columns([1,1.5], gap="small") # Classifiation Columns
            # Display uploaded image with limited size
            with col1:
                with st.spinner('Predicting...'):
                    st.image(image, caption='Uploaded/Captured Image', use_container_width='true',)  # Limit width            

                # Load selected model
                model = self.models[selected_model] # This is for preloading
                # model = load_model(selected_model)
            # Classify image
            with col2:
                if model:
                    predictions = classify_image(
                        image, 
                        model, 
                        confidence_threshold,
                        top_k
                    )
                    
                    # Display results in an expandable section (accordion)
                    if predictions:
                        # Load cached datasets
                        datasets_by_scientific_name = {item['meta']['scientific_name']: item for item in self.tree_datasets}
                        top_prediction = predictions[0]
                        with st.expander("Top Prediction", expanded=True):
                            st.subheader(top_prediction['common_name'])
                            # Info button for more details about the common name
                            info_button_key = f"info_{top_prediction['common_name']}"
                            if st.button("ℹ️ Show details...", key=info_button_key, use_container_width=True):
                                scientific_name = top_prediction['label']
                                if scientific_name in datasets_by_scientific_name:
                                    show_dataset_details(datasets_by_scientific_name[scientific_name])
                            st.metric(
                                label=top_prediction['label'], 
                                value=f"{top_prediction['score']:.2%}",
                                border=True
                            )
                        if len(predictions) > 1:  # If there are more predictions available
                            with st.expander("View More Predictions"):
                                self.display_predictions(predictions[1:], datasets_by_scientific_name)
                    else:
                        st.warning("No predictions above the confidence threshold")
    
    def display_predictions(self, predictions, datasets_by_scientific_name, max_classes=9):
        """Display predictions in a two-column layout with info buttons linked to dataset details."""


        num_cols = 2
        num_rows = min(max_classes, len(predictions)) // num_cols
        remaining_preds = len(predictions) % num_cols

        # Render rows and columns of predictions
        for i in range(num_rows):
            col1, col2 = st.columns(num_cols)
            
            # Render first prediction
            pred1 = predictions[i * num_cols]
            self.render_prediction(pred1, col1, datasets_by_scientific_name)

            # Render second prediction if available
            if i * num_cols + 1 < len(predictions):
                pred2 = predictions[i * num_cols + 1]
                self.render_prediction(pred2, col2, datasets_by_scientific_name)

        # Handle remaining predictions if any (less than num_cols)
        if remaining_preds > 0:
            remaining_cols = st.columns(remaining_preds)
            for i in range(remaining_preds):
                pred = predictions[num_rows * num_cols + i]
                self.render_prediction(pred, remaining_cols[i], datasets_by_scientific_name)

    def render_prediction(self, pred, col, datasets_by_scientific_name):
        with col:
            # Create a container with a border
            container = st.container(border=True)
            with container:
                # Create a header with an info icon on the right
                st.subheader(pred['common_name'])
            
                # Info button for more details about the common name
                info_button_key = f"info_{pred['common_name']}"
                if st.button("ℹ️ Show details...", key=info_button_key, use_container_width=True):
                    scientific_name = pred['label']
                    if scientific_name in datasets_by_scientific_name:
                        show_dataset_details(datasets_by_scientific_name[scientific_name])
                
                # Show metric with label and score
                st.metric(label=pred['label'], value=f"{pred['score']:.2%}", border=True)

    def datasets_tab(self):
        """Main function to display datasets in a grid layout."""
        st.title("Model Datasets")

        # Load cached datasets
        dataset_template = self.tree_datasets

        # Create columns for dataset display
        cols = st.columns(3)  # Use 3 columns for better layout

        # Display dataset cards
        for i, dataset in enumerate(dataset_template):
            with cols[i % 3]:  # Distribute datasets across columns
                with st.container():
                    # Display image
                    st.image(os.path.join('static/images/datasets', dataset['image']['path']), 
                            use_container_width=True)

                    # Show dialog on button click
                    if st.button(f"**{dataset['title']}**", key=f"dialog_btn_{i}", use_container_width=True):
                        show_dataset_details(dataset)

    def run(self):
        """Main app runner with sidebar control."""
        # Create tabs
        listTabs = ["Inference", "Datasets", "About"]
        whitespace = 9

        # Center and create tabs
        centered_tabs = [s.center(whitespace, "\u2001") for s in listTabs]
        tabs = st.tabs(centered_tabs)
        
        # Tab logic and session state tracking
        for i, tab in enumerate(tabs):
            with tab:
                st.session_state["active_tab"] = listTabs[i]
                if listTabs[i] == "Inference":
                    self.inference_tab()
                elif listTabs[i] == "Datasets":
                    self.datasets_tab()
                elif listTabs[i] == "About":
                    self.introduction_tab()
            
        # Sidebar content based on the active tab
        if st.session_state.get("active_tab") == "Inference":
            model_descriptions = {
                "ResNet-50": "ResNet-50: A convolutional neural network for image classification.",
                "ViT Base": "ViT Base: A vision transformer that uses self-attention for image classification.",
                "EfficientNet": "EfficientNet: A highly efficient CNN model optimized for accuracy and speed.",
                "Swin Base": "Swin Base: A vision transformer optimized for hierarchical representation.",
            }
            st.sidebar.markdown(
                f"**Selected Model:** {model_descriptions.get(st.session_state.get('current_model', 'ResNet-50'))}"
            )
