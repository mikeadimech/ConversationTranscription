import streamlit as st
from pages import utils
import time
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt

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
    script="Conversation Transcript\n"
    for index, row in df.iterrows():
        temp = "\nSpeaker " + row["Speaker"] + ":\n___________\n" + row["Transcript"] + "\n"
        script += temp
    return script

def summary_text(summary):
    temp_summary = "Meeting Summary:\n\n"
    temp_summary += summary
    return temp_summary

def app():
    
    with st.spinner("Loading sample meeting..."):
        audio, sr = read_display_audio("audio\sample_meeting.wav")
    with st.spinner("Detecting speakers..."):
        #dia_df = utils.diarization(st.session_state.audio_path)
        dia_df = utils.diarization_v3(audio,sr)
    st.write("Total Speakers: ",len(set(dia_df["Speaker"].to_list())))
    header = st.empty()
    with st.spinner("Generating transcript..."):
        transcript_df = utils.generate_transcript_dia_batches_v3(audio, sr, dia_df)
    
    header.markdown("### Transcript")
    transcript_text = st.empty()
    if transcript_df is not None:
        script_md = transcript_md(transcript_df)
        transcript_text.markdown(script_md)
        with st.spinner("Optimising transcript..."):
            optimised_transcript_df = utils.optimise_transcript(transcript_df)
        script_md = transcript_md(optimised_transcript_df)
        transcript_text.empty()
        transcript_text.markdown(script_md)
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
        
        summary_button_placeholder = st.empty()
        
        summary_output = st.empty()
        
        with st.spinner("Generating summary..."):
            summary_button_placeholder.empty()
            summary = utils.gen_summary(transcript_df)
        summary_output.markdown("### Summary\n"+summary)
        
        summ_txt = summary_text(summary)
        st.download_button(
            label="⬇️ Download Summary",
            data=summ_txt,
            file_name="Conversation Transcript.txt",
            mime='text/plain'
        )