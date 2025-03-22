import os

import streamlit as st
from PIL import Image

from modules.paths import STATIC_PATH_IMAGE
from modules.dataset import load_datasets

def show_welcome_view():
    """Display the welcome view with two columns layout and confirmation button"""
    # Container for welcome view
    with st.container():
        # Create two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Title with custom styling
            st.markdown("""
                <h1 style='
;'>
                    Barkley: Tree Bark Image Classification
                </h1>
            """, unsafe_allow_html=True)
            
            # Introduction text
            st.markdown("""
                <p style='
                    font-size: 1.1em;
                    line-height: 1.6;
                    color: var(--text-color);
                    margin-bottom: 1.5em;'>
                    Explore the fascinating world of trees through their unique bark patterns. Whether you're a botanist, researcher, or nature enthusiast, Barkley helps you identify tree species with just a photo. 
                    Nestled within the serene surroundings of the First Asia Institute of 
                    Technology and Humanities Colleges, our project sheds light on the 
                    distinctive barks of 13 indigenous species prevalent in this locale.
                </p>
            """, unsafe_allow_html=True)
            
            # Species lists
            species_items, _, _ = load_datasets()
            species_lists = "".join([f"<li>{item['title']}</li>" for item in species_items[:4]])
            species_lists += "<li>and more...</li>"

            html_content = f"""
                <p style='color: var(--text-color); margin-bottom: 0.5em;'>These encompass:</p>
                <ul style='
                    list-style-type: disc;
                    margin-left: 2em;
                    color: var(--text-color);
                    line-height: 1.8;
                    '>
                    {species_lists}
                </ul>
            """
            st.markdown(html_content, unsafe_allow_html=True)

            # Add some space before the button
            st.write("")
            st.write("")
            

            # Confirmation checkbox and button
            understand = st.checkbox(label="**I understand the purpose of this application**")

            if understand:
                    st.session_state.welcome_shown = True
                    st.rerun()
        
        with col2:
            # Load and display the image with some styling
            welcome_image = Image.open(os.path.join(STATIC_PATH_IMAGE, 'preview_placeholder.jpg'))
            st.image(
                welcome_image,
                use_container_width=True,
            )