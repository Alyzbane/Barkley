import streamlit as st
from modules.base import ImageClassification

if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="Barkley",
        initial_sidebar_state="collapsed",
        page_icon="🌴",
        layout="wide"
    )
    # Initialize and run app
    app = ImageClassification()

    app.run()