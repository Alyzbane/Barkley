import os

import streamlit as st
from PIL import Image

from modules.paths import STATIC_PATH_IMAGE

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
                    Nestled within the serene surroundings of the First Asia Institute of 
                    Technology and Humanities Colleges, our project sheds light on the 
                    distinctive barks of five indigenous species prevalent in this locale.
                </p>
            """, unsafe_allow_html=True)
            
            # Species list with styling
            st.markdown("""
                <p style='color: var(--text-color); margin-bottom: 0.5em;'>These encompass:</p>
                <ul style='
                    list-style-type: disc;
                    margin-left: 2em;
                    color: var(--text-color);
                    line-height: 1.8;'>
                    <li><i>Roystonea regia</i></li>
                    <li><i>Cananga odorata</i></li>
                    <li><i>Mangifera indica</i></li>
                    <li><i>Tabebuia</i></li>
                    <li><i>Pterocarpus indicus</i></li>
                </ul>
            """, unsafe_allow_html=True)
            
            # Add some space before the button
            st.write("")
            st.write("")
            

            # Confirmation checkbox and button
            understand = st.checkbox("I understand the purpose of this application")

            if understand:
                    st.session_state.welcome_shown = True
                    st.rerun()
        
        with col2:
            # Load and display the image with some styling
            welcome_image = Image.open(os.path.join(STATIC_PATH_IMAGE, 'preview_placeholder.webp'))
            st.image(
                welcome_image,
                use_container_width=True,
                caption="Tree Bark Collection"
            )




#################################################################################
#################################################################################
# def display_header():
#     """Display the centered header with title and description"""
#     # Using custom CSS to ensure text visibility in both light and dark themes
#     st.markdown("""
#         <style>
#         .title-container {
#             text-align: center;
#             padding: 5rem 0;
#         }
#         .main-title {
#             font-size: 4rem !important;
#             font-weight: bold !important;
#             margin-bottom: 1rem !important;
#             color: var(--text-color) !important;
#         }
#         .subtitle {
#             font-size: 2rem !important;
#             margin-bottom: 2rem !important;
#             color: var(--text-color) !important;
#         }
#         .description {
#             font-size: 1.25rem !important;
#             max-width: 30rem !important;
#             margin: 0 auto !important;
#             color: var(--text-color) !important;
#         }
#         /* Ensure text visibility in both themes */
#         [data-theme="light"] {
#             --text-color: #1E1E1E;
#         }
#         [data-theme="dark"] {
#             --text-color: #FFFFFF;
#         }
#         </style>
        
#         <div class="title-container">
#             <h1 class="main-title">Barkley</h1>
#             <h2 class="subtitle">Tree Bark Image Classification</h2>
#             <p class="description">
#                 Barkley can readily be used to identify the following barks: 
#                 <i>Roystonea regia</i>, <i>Cananga odorata</i>, <i>Mangifera indica</i>, 
#                 <i>Tabebuia</i> and <i>Pterocarpus indicus</i> from a single image.
#             </p>
#         </div>
#         """, unsafe_allow_html=True)
