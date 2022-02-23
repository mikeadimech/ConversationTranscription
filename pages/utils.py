import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
#from datasets import load_dataset
import soundfile as sf
import torch
import streamlit as st
from pyannote.audio import Pipeline, Inference
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from math import ceil
import openai

@st.cache(show_spinner=False)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def timestamp(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%06.3f" % (hour, minutes, seconds)

@st.experimental_singleton(show_spinner=False)
def load_W2V2():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    return processor, model

def dia(audio):
    model = Inference("pyannote/speaker-diarization")
    diarization = model(audio)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        st.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

def dia_pipeline(audio):
    #st.write("Loading pipeline")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    #st.write("pipeline loaded")
    # apply pretrained pipeline
    return pipeline(audio)

def asr_batches(speakers, durations):

    batches = []
    batch = 0
    max_batch_duration = 20
    temp_duration = 0

    for i, (speaker, duration) in enumerate(zip(speakers, durations)):
        
        #new speaker
        if batch == 0 or speaker != speakers[i-1]:
            #new batch
            temp_duration = duration
            batch+=1
            batches.append(batch)
        #continue prev speaker
        else:
            #if accumlulated duraction does not exceed max batch duration
            if (temp_duration+duration) < max_batch_duration:
                batches.append(batch)
                temp_duration+=duration
            #new batch
            else:
                temp_duration = duration
                batch+=1
                batches.append(batch)

    return batches

@st.cache(suppress_st_warning=True,show_spinner=False)
def diarization(audio):
    diarization = dia_pipeline(audio)

    temp_start = []
    temp_end = []
    temp_speaker = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        temp_start.append(turn.start)
        temp_end.append(turn.end)
        temp_speaker.append(chr(int(speaker.split("_",1)[1])+65))
    temp_duration = [x1 - x2 for (x1, x2) in zip(temp_end, temp_start)]
    temp_asr_batches = asr_batches(temp_speaker,temp_duration)
    return pd.DataFrame(list(zip(temp_start,temp_end,temp_speaker,temp_duration,temp_asr_batches)),columns=["Start","End","Speaker","Duration","ASR Batch"])

@st.cache(suppress_st_warning=True,show_spinner=False)
def generate_transcript_dia_batches_v2(audio, sr, dia_df):
    
    processor, model = load_W2V2()
    
    margin = int(0.1*sr)
    
    temp_speaker = "None"
    temp_transcript = ""
    transcripts = []
    speakers = []
    speaker_md = []
    t_md = []

    batches = dia_df["ASR Batch"].to_list()
    for batch in set(batches):
        temp_df = dia_df[dia_df["ASR Batch"]==batch]
        if temp_speaker == "None":
            #first speaker
            speaker_md.append(st.empty())
            speaker_md[-1].markdown("###### **Speaker "+temp_df["Speaker"].iloc[0]+"**")
            t_md.append(st.empty())

        elif temp_speaker != temp_df["Speaker"].iloc[0]:
            #new speaker
            t_md[-1].markdown("%s" % temp_transcript)
            transcripts.append(temp_transcript)
            speakers.append(temp_speaker)
            temp_transcript = ""
            speaker_md.append(st.empty())
            speaker_md[-1].markdown("###### **Speaker "+temp_df["Speaker"].iloc[0]+"**")
            t_md.append(st.empty())

        else:
            #continue prev speaker
            temp_transcript += " "
        
        temp_audio = audio[int(temp_df["Start"].iloc[0]*sr)-margin:int(temp_df["End"].iloc[0]*sr)+margin]
        for i in range(1,len(temp_df)):
            temp_audio = np.concatenate((temp_audio,audio[int(temp_df["Start"].iloc[i]*sr)-margin:int(temp_df["End"].iloc[i]*sr)+margin]))
        
        temp_transcript += asr(temp_audio,sr,processor,model)
        t_md[-1].markdown("%s..." % temp_transcript)
        temp_speaker = temp_df["Speaker"].iloc[0]

    t_md[-1].markdown("%s" % temp_transcript)
    transcripts.append(temp_transcript)
    speakers.append(temp_speaker)
    for s, t in zip(speaker_md,t_md):
        s.empty()
        t.empty()

    return pd.DataFrame(list(zip(speakers, transcripts)),columns=["Speaker","Transcript"])

@st.cache(suppress_st_warning=True,show_spinner=False)
def generate_transcript_dia_batches(audio, sr, dia_df):
    
    processor, model = load_W2V2()
    
    margin = int(0.1*sr)
    
    temp_speaker = "None"
    temp_transcript = ""
    transcripts = []
    speakers = []

    batches = dia_df["ASR Batch"].to_list()
    for batch in set(batches):
        temp_df = dia_df[dia_df["ASR Batch"]==batch]
        if temp_speaker == "None":
            #first speaker
            st.markdown("##### **Speaker "+temp_df["Speaker"].iloc[0]+"**")
            t = st.empty()

        elif temp_speaker != temp_df["Speaker"].iloc[0]:
            #new speaker
            t.markdown("%s" % temp_transcript)
            transcripts.append(temp_transcript)
            speakers.append(temp_speaker)
            temp_transcript = ""
            st.markdown("###### **Speaker "+temp_df["Speaker"].iloc[0]+"**")
            t = st.empty()

        else:
            #continue prev speaker
            temp_transcript += " "
        
        temp_audio = audio[int(temp_df["Start"].iloc[0]*sr)-margin:int(temp_df["End"].iloc[0]*sr)+margin]
        for i in range(1,len(temp_df)):
            temp_audio = np.concatenate((temp_audio,audio[int(temp_df["Start"].iloc[i]*sr)-margin:int(temp_df["End"].iloc[i]*sr)+margin]))
        
        temp_transcript += asr(temp_audio,sr,processor,model)
        t.markdown("%s..." % temp_transcript)
        temp_speaker = temp_df["Speaker"].iloc[0]

    t.markdown("%s" % temp_transcript)
    transcripts.append(temp_transcript)
    speakers.append(temp_speaker)

    return pd.DataFrame(list(zip(speakers, transcripts)),columns=["Speaker","Transcript"])

def calc_splits(audio_len,split_val,threshold):
    return int(ceil((audio_len-threshold)/split_val))

def asr(audio, sr, processor, model):
    
    split_val = 20
    split_threshold = 5
    if (len(audio)/sr)>(split_threshold+split_threshold):
        no_splits = calc_splits(len(audio)/sr,split_val,split_threshold)
        non_silent_interval = librosa.effects.split(audio, top_db=10, hop_length=1000)
        temp_transcript = ""
        mid_point = 0
        if no_splits < len(non_silent_interval):
            for i in range(no_splits):
                temp = int((len(non_silent_interval)/no_splits)*(i+1))-1
                if i < (no_splits-1):
                    temp_transcript += asr_transcript(audio[mid_point:int((non_silent_interval[temp][0]+non_silent_interval[temp-1][1])/2)],sr,processor,model)
                    temp_transcript += " "
                    mid_point = int((non_silent_interval[temp][0]+non_silent_interval[temp-1][1])/2)
                else:
                    temp_transcript += asr_transcript(audio[mid_point:],sr,processor,model)
            return temp_transcript   
        else: 
            return asr_transcript(audio,sr,processor,model)    
    else:
        return asr_transcript(audio,sr,processor,model)

def asr_transcript(audio, sr, processor, model):
    
    # tokenize
    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=sr).input_values  # Batch size 1
 
    # retrieve logits (non-normalised prediction values)
    logits = model(input_values).logits
 
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.lower()

def calc_tokens():
    return 5

def get_prompt(transcript_df):
    return "hi"

@st.cache(suppress_st_warning=True,show_spinner=False)
def optimise_transcript(transcript_df):
    prompt = get_prompt(transcript_df) #"Punctuate the following paragraphs:\n1."
    tokens = calc_tokens()
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        temperature=0,
        max_tokens=tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    time.sleep(3)
    return transcript_df

@st.cache(suppress_st_warning=True,show_spinner=False)
def gen_summary(transcript_df):
    return "This is a summary"
