import streamlit as st
from modules.base import clear_session_state
from modules.dataset import show_dataset_details

###################################################
##   Prediction result in 2 column layout        ##
###################################################   
def display_predictions(predictions, datasets_by_scientific_name, max_classes=9):
    """Display predictions in a two-column layout with info buttons linked to dataset details."""
    num_cols = 2
    num_rows = min(max_classes, len(predictions)) // num_cols
    remaining_preds = len(predictions) % num_cols

    # Render rows and columns of predictions
    for i in range(num_rows):
        col1, col2 = st.columns(num_cols)
        render_prediction(predictions[i * num_cols], col1, datasets_by_scientific_name)
        if i * num_cols + 1 < len(predictions):  # Check for second column
            render_prediction(predictions[i * num_cols + 1], col2, datasets_by_scientific_name)

    # Handle remaining predictions
    if remaining_preds > 0:
        remaining_cols = st.columns(remaining_preds)
        for i in range(remaining_preds):
            render_prediction(predictions[num_rows * num_cols + i], remaining_cols[i], datasets_by_scientific_name)

########################################################################
## Render the data of predictions and information of data in button   ##
########################################################################
def render_prediction(pred, col, datasets_by_scientific_name):
    """Render predictions data and information."""
    with col:
        container = st.container(border=True)
        with container:
            st.subheader(pred['common_name'])
            if st.button("ℹ️ Show details...", key=f"info_{pred['common_name']}", use_container_width=True):
                scientific_name = pred['label']
                if scientific_name in datasets_by_scientific_name:
                    show_dataset_details(datasets_by_scientific_name[scientific_name])
            # Display the score
            conf_thresh = st.session_state.confidence_threshold
            top_score_diff = pred['score'] - conf_thresh
            st.metric(label=pred['label'], value=f"{pred['score']:.2%}", delta=f"{top_score_diff:.2%}", border=True)

####################################################################
##  Main function to display the results (wraps Streamlit logic)  ##
####################################################################
def show_results_view():
    """Main function to display the results."""
    # Check session state for predictions and image
    if "predictions" not in st.session_state:
        st.error("No results to display. Please classify an image first.")
        st.stop()
    if "image" not in st.session_state:
        st.error("No image to display. Please classify an image first.")
        st.stop()

    # Retrieve predictions and image
    predictions = st.session_state.predictions
    image = st.session_state.image

    # Display layout
    st.title("Classification Results")
    image_col, predictions_col = st.columns([0.2, 0.8], gap='small')
    with image_col.container(border=True):
        st.image(image, use_container_width=True)
    # Check if there are any predictions to display
    if predictions:
        # Load cached datasets (example code, adjust as per your app)
        # Placeholder metadata for scientific name details
        datasets_by_scientific_name = {
            item['meta']['scientific_name']: item for item in st.session_state.tree_datasets
        }
        # Top prediction
        top_prediction = predictions[0]
        with predictions_col.expander("Top Prediction", expanded=True):
            st.subheader(top_prediction['common_name'])
            
            # Info button for more details about the common name
            info_button_key = f"info_{top_prediction['common_name']}"
            if st.button("ℹ️ Show details...", key=info_button_key, use_container_width=True):
                scientific_name = top_prediction['label']
                if scientific_name in datasets_by_scientific_name:
                    show_dataset_details(datasets_by_scientific_name[scientific_name])
            
            # Display the score
            conf_thresh = st.session_state.confidence_threshold
            top_score_diff = top_prediction['score'] - conf_thresh
            st.metric(label=top_prediction['label'], value=f"{top_prediction['score']:.2%}", delta=f"{top_score_diff:.2%}", border=True)
        
        # Display additional predictions if available
        if len(predictions) > 1:
            with predictions_col.expander("View More Predictions"):
                display_predictions(predictions[1:], datasets_by_scientific_name)
    else:
        predictions_col.error("No predictions above the confidence threshold")
    if image_col.button('Try Another Image', use_container_width=True, type="primary"):
        clear_session_state()
        st.switch_page('app.py')

if __name__ == "__main__":
        # Set the page configuration
    st.set_page_config(
        page_title="Barkley Results",  # Title displayed in the browser tab
        page_icon="🌳",  # Icon displayed in the browser tab
        layout="wide",  # Optional: Sets layout to wide or centered
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