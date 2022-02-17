import streamlit as st
from pages import utils
import os
from pydub import AudioSegment


def upload_file():
    
    uploaded_file = st.file_uploader(label="", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"])
    if uploaded_file is not None:

        if uploaded_file.name.endswith('wav'):
            audio = AudioSegment.from_wav(uploaded_file)
            file_type = 'wav'
        elif uploaded_file.name.endswith('mp3'):
            audio = AudioSegment.from_mp3(uploaded_file)
            file_type = 'mp3'

        st.success("Uploaded "+uploaded_file.name)
        sound = AudioSegment.from_wav(uploaded_file)
        audio_path = os.getcwd()+"\\audio\\"+uploaded_file.name
        sound.export(audio_path,format=file_type)
        st.session_state["audio_file"] = uploaded_file
        st.audio(uploaded_file)
        
        with st.spinner("Generating Transcript"):
            return utils.generate_transcript_from_file(audio)

def app():
        
    text = upload_file()
    if text is not None:
        st.write(text)

        