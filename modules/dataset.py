import os
import json
import streamlit as st

from .utils import image_to_base64, load_json
from .paths import STATIC_PATH_IMAGE, STATIC_PATH_CSS

@st.cache_data(ttl=60*60)  # Cache expires after 1 hour
def load_datasets():
    """Load dataset information from JSON and cache it."""
    gist_ds = 'https://gist.githubusercontent.com/Alyzbane/2fbbfdbc1da55e4952d501e8371de02f/raw/datasets'
    try:
        data = load_json(url=gist_ds) # Using the gist file to laod the datasets information
        items = data['items']

        scientific_to_common = {}
        common_to_scientific = {}

        for item in items:
            scientific_name = item['meta']['scientific_name']
            primary_common_name = item['title']  # Use title as common name
            
            # Store mappings
            scientific_to_common[scientific_name] = primary_common_name
            common_to_scientific[primary_common_name] = scientific_name

        return items, scientific_to_common, common_to_scientific

    except FileNotFoundError:
        st.error("Dataset JSON file not found!")
    except json.JSONDecodeError:
        st.error("Invalid JSON format in dataset file!")
    except KeyError:
        st.error("JSON file missing 'items' key!")

    return [], {}, {}

def view_datasets(dataset, confidence=None):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
            <div class='detail-header'>
                <h1>{dataset['title']}</h1>
                <p><strong>Scientific Name:</strong> {dataset['meta']['scientific_name']}</p>
                <p><strong>Common Names:</strong> {', '.join(dataset['meta']['common_names'])}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<p>{dataset['description']}</p>", unsafe_allow_html=True)
        
        # Show confidence in Tab 2 if provided
        if confidence is not None:
            st.markdown(f"""
            <div class='confidence-container'>
                <div class='confidence-label'>Confidence:</div>
                <div class='confidence-bar'>
                    <div class='confidence-fill' style='width: {confidence*100}%;'></div>
                </div>
                <div class='confidence-percentage'>{confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.image(
            os.path.join(STATIC_PATH_IMAGE, 'datasets', dataset['image']['path']), 
            use_container_width=True
        )
    
    st.markdown("### Characteristics")
    cols = st.columns(3)
    characteristics = [
        ("Geographic Location", dataset['characteristics']['geographic_location']),
        ("Height", dataset['characteristics']['height']),
        ("Trunk Diameter", dataset['characteristics']['trunk_diameter']),
        ("Color", ', '.join(dataset['characteristics']['color'])),
        ("Texture", ', '.join(dataset['characteristics']['texture']))
    ]
    
    for i, (label, value) in enumerate(characteristics):
        with cols[i % 3]:
            st.markdown(f"""
                <div class='characteristic-card'>
                    <div class='label'>
                        <strong>{label}</strong>
                        <div class='value'>
                            {value}
                        </div>
                    </div>

                </div>
            """, unsafe_allow_html=True)

@st.dialog("Dataset Details", width='large')
def dialog_dataset_info(dataset, confidence=None):
    with open(os.path.join(STATIC_PATH_CSS, "result.css")) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    view_datasets(dataset, confidence)

@st.cache_data(ttl=60*60, max_entries=1) 
def prepare_dataset_cards(dataset_template):
    """Process dataset items into card data."""
    card_data_list = []
    for dataset in dataset_template:
        image_path = os.path.join('static/images/datasets', dataset['image']['path'])
        with open(image_path, 'rb') as img_file:
            image_data = image_to_base64(os.path.join('static/images/datasets', dataset['image']['path']))
        card_data = {
            'title': dataset['title'],
            'image_data': image_data,
            'dataset': dataset
        }
        card_data_list.append(card_data)
    return card_data_list

def datasets_tab():
    """Main function to display datasets in a grid layout."""
    dataset_template = st.session_state.tree_datasets
    card_data_list = prepare_dataset_cards(dataset_template)

    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.5;">
                Below, you will find a collection of datasets that were utilized during the training process of our model. 
                Each dataset contains valuable information that contributed to the development and accuracy of the model.
                Click "Learn More" on any dataset to explore.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with open(os.path.join(STATIC_PATH_CSS, "dataset.css")) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, card_data in enumerate(card_data_list):
        with cols[i % 5]:
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <img src="data:image/jpeg;base64, {card_data['image_data']}" alt="{card_data['title']}">
                    <h3>{card_data['title']}</h3>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Learn More", key=f"dialog_btn_{i}", use_container_width=True):
                    dialog_dataset_info(card_data['dataset'])
