import os
import streamlit as st
from streamlit_option_menu import option_menu

from modules.inference import inference_tab
from modules.dataset import datasets_tab
from modules.team import team_tab
from modules.paths import STATIC_PATH_CSS

from views.welcome import show_welcome_view

def init_session_state():
    """Initialize session state variables."""
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "ResNet-50"
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.5
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    if 'tree_datasets' not in st.session_state:
        from modules.dataset import load_datasets
        st.session_state.tree_datasets = load_datasets()
    if 'welcome_shown' not in st.session_state:
        st.session_state.welcome_shown = False

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Barkley",
        initial_sidebar_state="collapsed",
        page_icon="🌳",  
        layout="wide"
    )

     # Initialize session state
    init_session_state()

    with open(os.path.join(STATIC_PATH_CSS, "style.css")) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    if not st.session_state.welcome_shown:
        show_welcome_view()
    else:
        # Create option menu without displaying icons
        selected = option_menu(
            menu_title=None,
            options=["Home", "Datasets", "Team"],
            icons=["house-fill", "tree-fill",  "people-fill"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent", "width": "100%"},
                "nav": {"display": "flex", "justify-content": "space-around", "width": "100%"},
                "nav-item": {"flex": "1"},
                "nav-link": {
                    "width": "100%",
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                    "padding": "10px 0",
                    "background-color": "transparent",
                    "border": "none",
                    "color": "var(--text-color)",
                    "font-weight": "bold",
                },
                "nav-link-selected": {
                    "color": "#166534",  # Highlight color for text and icon
                    "font-weight": "bold",  # Highlight color for text and icon

                },
                # "icon": {"display": "none"}  # Hide the icons
            }
        )
        
        # Handle tab content
        if selected == "Home":
            inference_tab()
        elif selected == "Datasets":
            datasets_tab()
        elif selected == "Team":
            team_tab()

if __name__ == "__main__":
    main()
