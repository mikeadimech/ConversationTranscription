from turtle import color
import streamlit as st
from pages import utils
import os
from pydub import AudioSegment
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio

from speechbrain.pretrained import EncoderClassifier

def upload_file():
    
    uploaded_file = st.file_uploader(label="", type=[".wav", ".mp3", ".flac", ".ogg"])
    if uploaded_file is not None:

        #if uploaded_file.name.endswith('wav'):
            #audio = AudioSegment.from_wav(uploaded_file)
            #file_type = 'wav'
        #elif uploaded_file.name.endswith('mp3'):
            #audio = AudioSegment.from_mp3(uploaded_file)
            #file_type = 'mp3'

        st.success("Uploaded "+uploaded_file.name)
        sound = AudioSegment.from_wav(uploaded_file)
        audio_path = os.getcwd()+"\\audio\\"+uploaded_file.name
        sound.export(audio_path,format="wav")
        st.session_state["audio_path"] = audio_path
        
        return uploaded_file
    return None

def read_display_audio(uploaded_file):
    st.audio(uploaded_file)
    audio, sr = librosa.load(uploaded_file,sr=16000)
    fig = plt.figure(figsize=(10,1), dpi=150)
    plt.axis('off')
    plt.rcParams['axes.facecolor']='none'
    plt.rcParams['savefig.facecolor']='none'
    librosa.display.waveshow(audio, sr=sr, color="white")
    st.pyplot(fig)
    return audio, sr

def app():
    
    uploaded_file = upload_file()
    if uploaded_file is not None:
        audio, sr = read_display_audio(uploaded_file)
        with st.spinner("Detecting speakers..."):
            dia_df = utils.diarization(st.session_state.audio_path)
        with st.spinner("Generating transcript..."):
            transcript_df = utils.generate_transcript_dia(audio, sr, dia_df)
        if transcript_df is not None:
            st.markdown("#### Transcript")
            st.dataframe(transcript_df)

        