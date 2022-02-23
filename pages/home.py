import streamlit as st
from pages import utils
import torch

def app():
    st.legacy_caching.clear_cache()
    