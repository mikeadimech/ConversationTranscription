import streamlit as st
import os
from pydub import AudioSegment
import wavio as wv
import time


st.set_page_config(
     page_title="Automatic Conversation Transcription",
     page_icon="🎙️",
     #layout="wide",
 )
st.title("🎙️ Automatic Conversation Transcription")

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
        sound.export(os.getcwd()+"\\audio\\"+uploaded_file.name,format="wav")
        st.audio(uploaded_file)
        with st.spinner("Generating Transcript"):
            return generate_transcript()

def generate_transcript():
    time.sleep(5)
    return "teeeeeeeeeeext"

#if st.button("Upload"):
text = upload_file()
st.write(text)

