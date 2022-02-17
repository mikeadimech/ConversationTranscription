import streamlit as st
from pages import utils


def app():
    

    if "audio_file" not in st.session_state:
        st.session_state["audio_file"] = None
    else:
        st.audio(st.session_state.audio_file)


 

    