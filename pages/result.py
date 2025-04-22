import os
import streamlit as st

from modules.inference import clear_session_state
from modules.paths import STATIC_PATH_CSS
from modules.dataset import view_datasets

####################################################################
##  Main function to display the results (wraps Streamlit logic)  ##
####################################################################
def show_results_view():
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "image" not in st.session_state:
        st.session_state.image = None

    if not st.session_state.predictions or not st.session_state.image:
        st.error("No results or image to display. Please classify an image first.")
        if st.button("Return", type="primary"):
            st.switch_page("app.py")
    else:

        predictions = st.session_state.predictions
        image = st.session_state.image

        st.title("Classification Results")
        image_col, predictions_col = st.columns([0.3, 0.7], gap='small')

        with open(os.path.join(STATIC_PATH_CSS, "result.css")) as f:
            st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        with image_col:
            st.image(image, use_container_width=True)
        
        with predictions_col:
            if predictions:
                ds_scientific_name = {
                    item['meta']['scientific_name']: item for item in st.session_state.tree_datasets
                }
                
                tab1, tab2 = st.tabs(["**Prediction Results**", "**Additional Results**"])
                
                with tab1:
                    # Display top 1 prediction 
                    top_pred = predictions[0]
                    view_datasets(
                        ds_scientific_name[top_pred['scientific_name']],
                        confidence=top_pred['score']
                    )

                with tab2:
                    if len(predictions) > 2:
                        # Display additional predictions 
                        for pred in predictions[1:]:
                            with st.expander(label=f"**{pred['common_name']}**"):
                                view_datasets(
                                    ds_scientific_name[pred['scientific_name']],
                                    confidence=pred['score']
                                )
                    else:
                        st.info("No additional results")

        if image_col.button('Try Another Image', use_container_width=True, type="primary"):
            clear_session_state()
            st.switch_page('app.py')

if __name__ == "__main__":
    # Set the page configuration
    st.set_page_config(
        page_title="Barkley - Results",
        initial_sidebar_state="collapsed",
        page_icon="🌳",
        layout="wide",
    )

    st.markdown(
        """
    <style>
        [data-testid="stSidebarCollapsedControl"] {
            display: none
        }
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    show_results_view()