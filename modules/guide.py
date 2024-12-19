import streamlit as st
import os

from .paths import STATIC_PATH_IMAGE, STATIC_PATH_CSS

def guidelines_classification():
        """Display help dialog with formatted layout."""
        with open(os.path.join(STATIC_PATH_CSS, "guide.css")) as f:
                st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

        # Centered heading text with neat format
        st.markdown("<h2>Guidelines: Tree bark classification</h2>", unsafe_allow_html=True)
        st.divider()
        # Layout: Singular, clear object
        for correct, wrong, correct_caption, wrong_caption in [
                ("correct_01.png", "wrong_01.png", "Singular and clear tree bark", "Blurry image"),
                ("correct_02.png", "wrong_02.png", "Minimal background noise", "Cluttered scenes"),
                ("correct_03.png", "wrong_03.png", "Sharp focus", "Small or distant tree bark"),
        ]:
                # Each tip in a separate container
                with st.container(border=True):
                        # Columns for correct and wrong image pair
                        col1, col2, col3, col4 = st.columns([2, 4, 4, 2], gap='large')

                        # Correct image
                        with col2.container():
                                st.image(os.path.join(STATIC_PATH_IMAGE, "guide", correct), use_container_width=True)
                                st.markdown(f"<div class='image-caption'>{correct_caption}</div>", unsafe_allow_html=True)

                        # Wrong image
                        with col3.container():
                                st.image(os.path.join(STATIC_PATH_IMAGE, "guide", wrong), use_container_width=True)
                                st.markdown(f"<div class='image-caption'>{wrong_caption}</div>", unsafe_allow_html=True)