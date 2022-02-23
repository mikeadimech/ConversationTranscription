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
import numpy as np


def upload_file():
    
    uploaded_file = st.file_uploader(label="", type=[".wav", ".mp3", ".flac", ".ogg"])
    if uploaded_file is not None:

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

@st.cache
def transcript_md(df):
    script=""
    for index, row in df.iterrows():
        temp = "##### **Speaker " + row["Speaker"] + "**\n" + row["Transcript"] + "\n"
        script += temp
    return script

@st.cache
def transcript_txt(df):
    script="Conversation Transcript\n\n"
    for index, row in df.iterrows():
        temp = "Speaker " + row["Speaker"] + ":\n___________\n" + row["Transcript"] + "\n"
        script += temp
    return script

def app():
    
    uploaded_file = upload_file()
    if uploaded_file is not None:
        audio, sr = read_display_audio(uploaded_file)
        with st.spinner("Detecting speakers..."):
            dia_df = utils.diarization(st.session_state.audio_path)
        st.write("Total Speakers: ",len(set(dia_df["Speaker"].to_list())))
        header = st.empty()
        with st.spinner("Generating transcript..."):
            transcript_df = utils.generate_transcript_dia_batches_v2(audio, sr, dia_df)
        
        header.markdown("### Transcript")
        transcript_text = st.empty()
        if transcript_df is not None:
            script_md = transcript_md(transcript_df)
            transcript_text.markdown(script_md)
            #with st.spinner("Optimising transcript..."):
                #transcript_df = utils.optimise_transcript(transcript_df)
            #script_md = transcript_md(transcript_df)
            #transcript_text.empty()
            #transcript_text.markdown(script_md)
            csv = utils.convert_df(transcript_df)
            st.download_button(
                label="💾 Export CSV",
                data=csv,
                file_name="Conversation Transcript.csv",
                mime='text/csv'
            )
            txt = transcript_txt(transcript_df)
            st.download_button(
                label="📄 Download TXT",
                data=txt,
                file_name="Conversation Transcript.txt",
                mime='text/plain'
            )
            
            #if st.button(label="📝 Generate Summary"):
                #st.session_state[""] = utils.gen_summary(transcript_df)
            #summary_output = st.empty()
            #summary_output.markdown("### Summary\n"+summary)
                


        