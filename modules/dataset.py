import os
import streamlit as st

from .utils import load_json
from .paths import STATIC_PATH_JSON, STATIC_PATH_IMAGE

@st.cache_data
def load_datasets():
    """Load dataset information from JSON and cache it."""
    return load_json(os.path.join(STATIC_PATH_JSON, 'datasets.json'))['items']

@st.cache_data
def get_image_path(path):
    return os.path.join('static/images/datasets', path)

st.cache_data
@st.dialog("Dataset Details", width='large')
def show_dataset_details(dataset):
    """Reusable dialog to show detailed dataset information."""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(os.path.join(STATIC_PATH_IMAGE, 'datasets', dataset['image']['path']), use_container_width=True)
        
    with col2:
        st.markdown(f"### {dataset['title']}")
        st.markdown(f"**Scientific Name:** {dataset['meta']['scientific_name']}")
        st.markdown(f"**Common Names:** {', '.join(dataset['meta']['common_names'])}")
        st.markdown(f"**Description:** {dataset['description']}")

    # Characteristics section
    st.markdown("### Characteristics")
    col3, col4, col5 = st.columns([3, 3, 2])

    with col3:
        st.metric("Geographic Location", dataset['characteristics']['geographic_location'])
        st.metric("Color", ', '.join(dataset['characteristics']['color']))

    with col4:
        st.metric("Height", dataset['characteristics']['height'])
        st.metric("Texture", ', '.join(dataset['characteristics']['texture']))

    with col5:
        st.metric("Trunk Diameter", dataset['characteristics']['trunk_diameter'])

def datasets_tab():
    """Main function to display datasets in a grid layout."""
    st.title("Model Datasets")

    # Load cached datasets
    dataset_template = st.session_state.tree_datasets

    # Create columns for dataset display
    cols = st.columns(5)  # Use 3 columns for better layout

    # Display dataset cards
    for i, dataset in enumerate(dataset_template):
        with cols[i % 5]:  # Distribute datasets across columns
            with st.container():
                # Display image
                st.image(os.path.join('static/images/datasets', dataset['image']['path']), 
                        use_container_width=True)

                # Show dialog on button click
                if st.button(f"**{dataset['title']}**", key=f"dialog_btn_{i}", use_container_width=True):
                    show_dataset_details(dataset)