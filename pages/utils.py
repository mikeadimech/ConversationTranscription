import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
#from datasets import load_dataset
import soundfile as sf
import torch
import streamlit as st
from pyannote.audio import Pipeline, Inference
import numpy as np
import pandas as pd

def timestamp(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%06.3f" % (hour, minutes, seconds)

def load_W2V2():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return processor, model

def dia(audio):
    model = Inference("pyannote/speaker-diarization")
    diarization = model(audio)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        st.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

def diarization(audio):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # apply pretrained pipeline
    diarization = pipeline(audio)

    temp_start = []
    temp_end = []
    temp_speaker = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        #st.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        temp_start.append(turn.start)
        temp_end.append(turn.end)
        temp_speaker.append(speaker)
    return pd.DataFrame(list(zip(temp_start,temp_end,temp_speaker)),columns=["Start","End","Speaker"])

def generate_transcript_dia(audio, sr, dia_df):
    
    transcripts = []
    for index, row in dia_df.iterrows():
        temp_string = asr(audio[int(row["Start"]*sr):int(row["End"]*sr)],sr)
        st.markdown("**"+row["Speaker"]+"** "+timestamp(row["Start"])+" - "+timestamp(row["End"])+" ",unsafe_allow_html=True)
        st.write(temp_string)
        transcripts.append(temp_string)
    dia_df["Transcript"] = transcripts
    return dia_df

def asr(audio, sr):
    
    processor, model = load_W2V2()

    # tokenize
    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=sr).input_values  # Batch size 1
 
    # retrieve logits (non-normalised prediction values)
    logits = model(input_values).logits
 
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription