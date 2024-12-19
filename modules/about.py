import os

import streamlit as st
from .paths  import STATIC_PATH_IMAGE

def about_tab():
    """About tab content"""
    st.title("Barkgods")
    left_column, center_column, right_column = st.columns(3)
    with center_column:
        st.image(os.path.join(STATIC_PATH_IMAGE, "chudda.jpg"), use_container_width=True)