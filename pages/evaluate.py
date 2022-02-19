import streamlit as st
from pages import utils


def app():
    st.metric(label="WER", value="1.4%", delta="5%")