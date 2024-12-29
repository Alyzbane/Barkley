import streamlit as st
import os

from modules.paths import STATIC_PATH_IMAGE, STATIC_PATH_CSS

def display_guidelines():
        """Display help dialog with formatted layout."""
        with open(os.path.join(STATIC_PATH_CSS, "guide.css")) as f:
                st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

        # Centered heading text with neat format
        st.markdown("<h4>HOW TO TAKE A PICTURE FOR TREE BARK CLASSIFICATION</h2>", unsafe_allow_html=True)
        st.divider()
        # Layout: Singular, clear object
        for correct, wrong, correct_caption, wrong_caption in [
                ("correct_01.jpg", "wrong_01.jpg", "Singular and clear tree bark", "Blurry image"),
                ("correct_02.jpg", "wrong_02.jpg", "Minimal background noise", "Cluttered scenes"),
                ("correct_03.jpg", "wrong_03.jpg", "Sharp focus", "Small or distant tree bark"),
        ]:
                # Each tip in a separate container
                with st.container(border=True):
                        # Columns for correct and wrong image pair
                        col1, col2 = st.columns([0.2, 0.2], gap='large')

                        # Correct image
                        with col1.container():
                                st.image(os.path.join(STATIC_PATH_IMAGE, "guide", correct), use_container_width=True)
                                st.markdown(f"<div class='image-caption'>{correct_caption}</div>", unsafe_allow_html=True)

                        # Wrong image
                        with col2.container():
                                st.image(os.path.join(STATIC_PATH_IMAGE, "guide", wrong), use_container_width=True)
                                st.markdown(f"<div class='image-caption'>{wrong_caption}</div>", unsafe_allow_html=True)