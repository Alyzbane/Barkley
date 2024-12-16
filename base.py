import os
import json
import time
import requests
from io import BytesIO

from PIL import Image
import streamlit as st

from model import load_model
from utils import load_json, resize_image, select_image

class ImageClassification:
    def __init__(self):
        # Initialize available models
        self.models = {
            "ResNet-50": "models/ResNet-50",
            "ViT Base": "models/ViT-base-patch16",
            "ConvNeXT": "models/ConvNeXT",
            "Swin Base": "models/Swin-base-patch4-window7",
        }

        # Initialize session state
        if 'current_model' not in st.session_state:
            st.session_state.current_model = list(self.models.keys())[0]
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 0.5
        if 'top_k' not in st.session_state:
            st.session_state.top_k = 5


    def classify_image(self, image, model, confidence_threshold, top_k):
        """Classify image with confidence filtering and name mapping"""
        try:
            # Perform classification
            predictions = model(image, top_k=top_k)

            # Define the mapping of scientific names to common names
            name_mapping = {
                "Roystonea regia": "Royal Palm",
                "Pterocarpus indicus": "Narra",
                "Tabebuia": "Trumpet Tree",
                "Mangifera indica": "Mango Tree",
                "Iinstia bijuga": "Ilang-ilang Tree"
            }

            # Filter predictions based on confidence threshold and update names
            filtered_predictions = []
            for pred in predictions:
                if pred['score'] >= confidence_threshold:
                    scientific_name = pred['label']
                    
                    # Update scientific name if necessary
                    if scientific_name == "Iinstia bijuga":
                        scientific_name = "Cananga odorata"
                    
                    # Get the common name from the mapping
                    common_name = name_mapping.get(pred['label'], "Unknown")
                    
                    # Update the prediction dictionary
                    pred['scientific_name'] = scientific_name
                    pred['common_name'] = common_name
                    
                    filtered_predictions.append(pred)

            return filtered_predictions
        except Exception as e:
            st.error(f"Classification error: {e}")
            return []

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
        st.title("Image Classification Inference")
        # Help Button
        if st.button("📘 How to Use"):
            self.show_help_dialog()
            
        # Sidebar configuration (remains the same as previous implementation)
        with st.sidebar:
            st.header("Model Configuration")
            
            # Model selection
            selected_model = st.selectbox(
                "Select Classification Model", 
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
                image, _ = select_image()


        # Image Classification Process
        if image:
            # Loading animation during inference
            with st.spinner('Classifying image...'):
                time.sleep(1.5)  # Simulate processing time
            col1, col2 = st.columns([1,1.5], gap="small") # Classifiation Columns
            # Display uploaded image with limited size
            with col1:
                st.image(image, caption='Uploaded/Captured Image', use_container_width='true',)  # Limit width            
                
                # Load selected model
                model = load_model(selected_model)
                
                if model:
                    # Classify image
                    with col2:
                        predictions = self.classify_image(
                            image, 
                            model, 
                            confidence_threshold,
                            top_k
                        )
                        
                        # Display results in an expandable section (accordion)
                        if predictions:
                            top_prediction = predictions[0]
                            with st.expander("Top Prediction", expanded=True):
                                st.subheader(top_prediction['common_name'])
                                st.metric(
                                    label=top_prediction['label'], 
                                    value=f"{top_prediction['score']:.2%}",
                                    border=True
                                )
                            if len(predictions) > 1:  # If there are more predictions available
                                with st.expander("View More Predictions"):
                                    self.display_predictions(predictions[1:])
                        else:
                            st.warning("No predictions above the confidence threshold")


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

    def display_predictions(self, predictions, max_classes=9):
        """Display predictions in a two-column layout with max_classes rows"""
        num_cols = 2
        num_rows = min(max_classes, len(predictions)) // num_cols
        remaining_preds = len(predictions) % num_cols

        for i in range(num_rows):
            col1, col2 = st.columns(num_cols)
            pred1 = predictions[i * num_cols]
            col1.subheader(pred1['common_name'])
            col1.metric(label=pred1['label'], value=f"{pred1['score']:.2%}", border=True)

            if i * num_cols + 1 < len(predictions):
                pred2 = predictions[i * num_cols + 1]
                col2.subheader(pred2['common_name'])
                col2.metric(label=pred2['label'], value=f"{pred2['score']:.2%}", border=True)

        # Handle remaining predictions if any (less than num_cols)
        if remaining_preds > 0:
            remaining_cols = st.columns(remaining_preds)
            for i in range(remaining_preds):
                pred = predictions[num_rows * num_cols + i]
                remaining_cols[i].subheader(pred['common_name'])
                remaining_cols[i].metric(label=pred['label'], value=f"{pred['score']:.2%}", border=True)


    def datasets_tab(self):
        st.title("Model Datasets")
        
        # Load datasets
        data = load_json('static/json/datasets.json')
        dataset_template = data['items']
        
        # Create columns for dataset display
        cols = st.columns(3)  # Use 3 columns for better layout

        @st.dialog("Dataset Details", width='large')
        def show_dataset_details(dataset):
            """Dialog to show detailed dataset information"""
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(os.path.join('static/images/datasets', dataset['image']['path']), 
                        use_container_width=True)
            
            with col2:
                st.markdown(f"### {dataset['title']}")
                st.markdown(f"**Scientific Name:** {dataset['meta']['scientific_name']}")
                st.markdown(f"**Common Names:** {', '.join(dataset['meta']['common_names'])}")
                st.markdown(f"**Description:** {dataset['description']}")
            
            # Characteristics section
            st.markdown("### Characteristics")
            col3, col4, col5 = st.columns([3, 3, 1])
            
            with col3:
                st.metric("Color", ', '.join(dataset['characteristics']['color']))
                st.metric("Geographic Location", dataset['characteristics']['geographic_location'])
            
            with col4:
                st.metric("Texture", ', '.join(dataset['characteristics']['texture']))
                st.metric("Height", dataset['characteristics']['height'])
            
            with col5:
                st.metric("Trunk Diameter", dataset['characteristics']['trunk_diameter'])
                
        # Create dataset cards
        for i, dataset in enumerate(dataset_template):
            with cols[i % 3]:  # Distribute datasets across columns
                with st.container():
                    st.image(os.path.join('static/images/datasets', dataset['image']['path']), 
                            use_container_width=True)
                    if st.button(f"**{dataset['title']}**", key=f"dialog_btn_{i}"):
                            show_dataset_details(dataset)

    def run(self):
        """Main app runner with sidebar control."""
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Introduction", "Inference", "Datasets"])

        # Tab logic
        with tab1:
            st.session_state["active_tab"] = "Introduction"
            self.introduction_tab()

        with tab2:
            st.session_state["active_tab"] = "Inference"
            self.inference_tab()

        with tab3:
            st.session_state["active_tab"] = "Datasets"
            self.datasets_tab()

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
