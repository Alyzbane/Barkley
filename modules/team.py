import os

import streamlit as st

from modules.paths import STATIC_PATH_IMAGE
from modules.utils import image_to_base64

def team_tab():
    # Sample team data with GitHub links
    team_members = [
        {
            "name": "Prince Daniel D. Mampusti",
            "role": "Full Stack ML Engineer",
            "image": "https://avatars.githubusercontent.com/u/75820221?s=400&u=55cf39b5f2cbfc565370a14303c6c12620ca6b6e&v=4", 
            "github": "https://github.com/alyzbane" 
        },
        {
            "name": "Dhan Eldrin Mabilangan",
            "role": "Data Analyst",
            "image": "https://static.wikia.nocookie.net/meme/images/9/9f/Dolan.png/revision/latest?cb=20160118072622", 
            "github": "https://github.com/dhan-eldrin"  
        },
        {
            "name": "Reymer Jr. Unciano",
            "role": "Data Curator",
            "image": "https://pbs.twimg.com/profile_images/1844554652207153152/gMB26r0n_400x400.jpg",  
            "github": "https://github.com/reymer-jr"  
        },
        {
            "name": "Tobias Alren",
            "role": "Data Collector",
            "image": "https://via.placeholder.com/150",  
            "github": "https://github.com/tobias-alren" 
        }
    ]

    # Title
    st.title("Meet Our Team")

    # Modern team layout using Streamlit columns
    cols = st.columns(len(team_members))



#    theme = st_theme() # buggy returning None not subscriptable
#    base_theme = theme.get("base", "default_value")
    base_theme = "light"

    if base_theme == "light":
        github_icon = image_to_base64(os.path.join(STATIC_PATH_IMAGE, 'logos', 'github-mark.png'))
    else:
        github_icon = image_to_base64(os.path.join(STATIC_PATH_IMAGE, 'logos', 'github-mark-white.png'))

    for idx, member in enumerate(team_members):
        with cols[idx]:
            st.markdown(f"""
            <div class="card" style="height: 350px; margin-bottom: 0.5em;">
                <img src="{member['image']}" alt="{member['name']}" width="100" height="100" style="align-self: center;">
                <h3 style="color: var(--text-color); text-align: center;">{member['name']}</h3>
                <a href="{member['github']}" target="_blank" style="display: block; text-align: center; margin-bottom: 10px;">
                    <img src="data:image/jpeg;base64,{github_icon}" alt="GitHub" width="30" height="30" />
                </a>
                <p style="color: ; text-align: center;">{member['role']}</p>
            </div>
            """, unsafe_allow_html=True)
