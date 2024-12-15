import streamlit as st
from PIL import Image
import time
import requests
from io import BytesIO
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import pipeline

class ImageClassificationApp:
    def __init__(self):
        # Initialize available models
        self.models = {
            "ResNet-50": "microsoft/resnet-50",
            "ViT Base": "google/vit-base-patch16-224",
            "EfficientNet": "google/efficientnet-b0"
        }

        # Initialize session state
        if 'current_model' not in st.session_state:
            st.session_state.current_model = list(self.models.keys())[0]
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 0.5
        if 'top_k' not in st.session_state:
            st.session_state.top_k = 5
            
    def load_model(self, model_name):
        """Load the selected model"""
        try:
            model_id = self.models[model_name]
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)
            
            # Create classification pipeline
            classifier = pipeline(
                "image-classification", 
                model=model, 
                feature_extractor=feature_extractor
            )
            return classifier
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def classify_image(self, image, model, confidence_threshold, top_k):
        """Classify image with confidence filtering"""
        try:
            # Perform classification
            predictions = model(image, top_k=top_k)
            
            # Filter predictions based on confidence threshold
            filtered_predictions = [
                pred for pred in predictions 
                if pred['score'] >= confidence_threshold
            ]
            
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

    def inference_tab(self):
        """Inference tab content for image classification"""
        st.title("Image Classification Inference")
        st.markdown("### How to Use This Application")

        with st.expander("📘 How to Use", expanded=False):
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
            ["Upload Image", "Camera Capture", "Paste Image URL"],
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
                camera_image = st.camera_input("Capture Image")
                image = self._process_uploaded_file(camera_image)

            elif upload_method == "Paste Image URL":
                image_url = st.text_input("Paste Image URL")
                if st.button("Submit URL"):
                    if image_url:
                        try:
                            response = requests.get(image_url)
                            image = Image.open(BytesIO(response.content))
                            image = self.resize_image(image, (224, 224))  # Resize to fit better in layout
                        except Exception as e:
                            st.error("Invalid URL or unable to download image")
                            image = None

        # Image Classification Process
        col1, col2 = st.columns([1,2,], gap="small") # Classifiation Columns
        
        if image:
            # Display uploaded image with limited size
            with col1:
                st.image(image, caption='Uploaded/Captured Image', use_container_width='true',)  # Limit width            
                
                # Load selected model
                model = self.load_model(selected_model)
                
                if model:
                    # Classify image
                    with col2:
                        # Loading animation during inference
                        with st.spinner('Classifying image...'):
                            time.sleep(1.5)  # Simulate processing time
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
                                st.metric(
                                    label=top_prediction['label'], 
                                    value=f"{top_prediction['score']:.2%}"
                                )

                            if len(predictions) > 1:  # If there are more predictions available
                                with st.expander("View More Predictions"):
                                    self.display_predictions(predictions[1:])
                        else:
                            st.warning("No predictions above the confidence threshold")



    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded file or camera input"""
        if uploaded_file is not None:
            # Open and resize image
            image = Image.open(uploaded_file)
            image = self.resize_image(image, (224, 224))
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
            col1.metric(label=pred1['label'], value=f"{pred1['score']:.2%}", border=True)

            if i * num_cols + 1 < len(predictions):
                pred2 = predictions[i * num_cols + 1]
                col2.metric(label=pred2['label'], value=f"{pred2['score']:.2%}", border=True)

        # Handle remaining predictions if any (less than num_cols)
        if remaining_preds > 0:
            remaining_cols = st.columns(remaining_preds)
            for i in range(remaining_preds):
                pred = predictions[num_rows * num_cols + i]
                remaining_cols[i].metric(label=pred['label'], value=f"{pred['score']:.2%}", border=True)



    def resize_image(self, image, size):
        """Resize image while maintaining aspect ratio"""
        image.resize(size, Image.LANCZOS)
        return image

    def datasets_tab(self):
        """Datasets tab with expandable card views"""
        st.title("Model Datasets")
        
        # Predefined dataset information template
        dataset_template = [
            {
                "name": "ImageNet-1k",
                "description": "Large-scale hierarchical image database",
                "classes": 1000,
                "images": "1.2 million",
                # "image_path": "path/to/imagenet_icon.png"
            },
            {
                "name": "CIFAR-10",
                "description": "Subset of 80 million tiny images dataset",
                "classes": 10,
                "images": "60,000",
                # "image_path": "path/to/cifar_icon.png"
            }
            # More datasets can be added here
        ]
        
        # Create bento-style grid of dataset cards
        cols = st.columns(3)
        for i, dataset in enumerate(dataset_template):
            with cols[i % 3]:
                with st.container():
                    # Placeholder for dataset image (you'd replace with actual image)
                    st.image(dataset['image_path'] if 'image_path' in dataset else 'https://via.placeholder.com/150', use_container_width=True)
                    st.write(f"**{dataset['name']}**")
                
                # Expandable card with dataset details
                with st.expander("More Details"):
                    st.write(f"Description: {dataset['description']}")
                    st.write(f"Number of Classes: {dataset['classes']}")
                    st.write(f"Total Images: {dataset['images']}")

    def run(self):
        """Main app runner"""
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Introduction", 
            "Inference", 
            "Datasets"
        ])
        
        with tab1:
            self.introduction_tab()
        
        with tab2:
            self.inference_tab()
        
        with tab3:
            self.datasets_tab()
        
        model_descriptions = {
            "ResNet-50": "ResNet-50: A convolutional neural network for image classification.",
            "ViT Base": "ViT Base: A vision transformer that uses self-attention for image classification.",
            "EfficientNet": "EfficientNet: A highly efficient CNN model optimized for accuracy and speed."
        }
        st.sidebar.markdown(f"**Selected Model:** {model_descriptions[st.session_state.current_model]}")
        
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Image Classification App",
        page_icon=":microscope:",
        layout="wide"
    )
    
    # Initialize and run app
    app = ImageClassificationApp()
    app.run()

if __name__ == "__main__":
    main()