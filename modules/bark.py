import streamlit as st

from modules.dataset import get_image_path
from modules.utils import image_to_base64

def datasets_tab():
    """Main function to display datasets in a grid layout."""
    st.title("Model Datasets")

    # Load cached datasets
    dataset_template = st.session_state.tree_datasets
   
    # Prepare the Base64 encoded images and titles
    image_paths = [f"data:image/jpeg;base64, {image_to_base64(get_image_path(dataset['image']['path']))}" for dataset in dataset_template]
    titles = [dataset['title'] for dataset in dataset_template]