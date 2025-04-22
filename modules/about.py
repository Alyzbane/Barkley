import os
import streamlit as st
from  .utils import image_to_base64, load_css

from modules.paths import STATIC_PATH_IMAGE

@st.cache_data(ttl=60*60)
def team_tab():
    team_members = [
        {
            "name": "Prince Daniel D. Mampusti",
            "role": "Full Stack Developer",
            "image": os.path.join(STATIC_PATH_IMAGE, 'team', 'img-mampusti.jpg'),
            "github": "https://github.com/alyzbane",
        },
        {
            "name": "Dhan Eldrin Mabilangan",
            "role": "Data Analyst",
            "image": os.path.join(STATIC_PATH_IMAGE, 'team', 'img-mabilangan.jpg'),
            "github": "https://github.com" 
        },
        {
            "name": "Reymer Jr. Unciano",
            "role": "Data Curator",
            "image": os.path.join(STATIC_PATH_IMAGE, 'team', 'img-unciano.jpg'),
            "github": "https://github.com"
        },
        {
            "name": "Tobias Alren",
            "role": "Data Collector",
            "image": os.path.join(STATIC_PATH_IMAGE, 'team', 'img-tobias.jpg'),
            "github": "https://github.com"
        }
    ]

    st.markdown("""
        <div>
            <h1 style="font-family: Arial, sans-serif;">Meet the team</h1>
        </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(team_members))
    load_css("card.css")

    base_theme = "light"
    if base_theme == "light":
        github_icon = image_to_base64(os.path.join(STATIC_PATH_IMAGE, 'logos', 'github-mark.png'))
    else:
        github_icon = image_to_base64(os.path.join(STATIC_PATH_IMAGE, 'logos', 'github-mark-white.png'))

    for idx, member in enumerate(team_members):
        with cols[idx]:
            img_base64 = image_to_base64(member["image"])
            st.markdown(f"""
            <div class="card" style="height: 350px; margin-bottom: 0.5em;">
                <img src="data:image/jpeg;base64,{img_base64}" 
                     alt="{member['name']}" 
                     width="100" height="100" 
                     style="display: block; margin-left: auto; margin-right: auto; border-radius: 50%;">
                <h3 style="color: var(--text-color); text-align: center;">{member['name']}</h3>
                <a href="{member['github']}" target="_blank" style="display: block; text-align: center; margin-bottom: 10px;">
                    <img src="data:image/jpeg;base64,{github_icon}" alt="GitHub" width="30" height="30" />
                </a>
                <p style="text-align: center;">{member['role']}</p>
            </div>
            """, unsafe_allow_html=True)

def about_tab():
    team_tab()
