import time
import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
#from datasets import load_dataset
import torch
import streamlit as st
from pyannote.audio import Pipeline, Inference
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from math import ceil
import openai
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.spatial.distance import cdist
import neuspell
import speechbrain as sb
import json
from scipy.io.wavfile import write
import requests
import ast

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
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    # apply pretrained pipeline
    return pipeline(audio)

def seg_pipeline(audio):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation")
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



def seg_to_dia_v2(audio,seg_output):
    model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))


    audio = Audio(sample_rate=16000, mono=True)

    speakers = {}
    spks=0

    temp_start = []
    temp_end = []
    temp_speaker = []

    for turn, _, speaker in seg_output.itertracks(yield_label=True):
        #print("\n",speaker,"\t",turn.start,"\t",turn.end)
        temp_start.append(turn.start)
        temp_end.append(turn.end)
        speaker = Segment(turn.start,turn.end)
        waveform, sample_rate = audio.crop(st.session_state.audio_path, speaker)
        embedding = model(waveform[None])
        if not bool(speakers):
            speakers["A"] = [embedding]
            #print("Speaker",chr(spks+65))
            temp_speaker.append("A")
        else:
            speaker_match = False
            for spk, emb in speakers.items():
                distances = []
                for e in emb:
                    distances.append(cdist(embedding, e, metric="cosine")[0,0])
                if not any([d>0.95 for d in distances]):
                    #speaker match
                    speaker_match = True
                    #print("Match found - speaker",spk,"\t",distance)
                    temp_speaker.append(spk)
                    speakers[spk].append(embedding)
                    break

            if not speaker_match:
                #new speaker
                spks+=1
                #print("new speaker",chr(spks+65))
                temp_speaker.append(chr(spks+65))
                speakers[chr(spks+65)] = [embedding]


    return temp_start, temp_end, temp_speaker


def seg_to_dia(audio,seg_output):
    model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))


    audio = Audio(sample_rate=16000, mono=True)

    speakers = {}
    spks=0

    temp_start = []
    temp_end = []
    temp_speaker = []

    for turn, _, speaker in seg_output.itertracks(yield_label=True):
        #print("\n",speaker,"\t",turn.start,"\t",turn.end)
        temp_start.append(turn.start)
        temp_end.append(turn.end)
        speaker = Segment(turn.start,turn.end)
        waveform, sample_rate = audio.crop(st.session_state.audio_path, speaker)
        embedding = model(waveform[None])
        if not bool(speakers):
            speakers["A"] = embedding
            #print("Speaker",chr(spks+65))
            temp_speaker.append("A")
        else:
            speaker_match = False
            for spk, emb in speakers.items():
                distance = cdist(embedding, emb, metric="cosine")[0,0]
                if not distance > 0.85:
                    #speaker match
                    speaker_match = True
                    #print("Match found - speaker",spk,"\t",distance)
                    temp_speaker.append(spk)
                    break

            if not speaker_match:
                #new speaker
                spks+=1
                #print("new speaker",chr(spks+65))
                temp_speaker.append(chr(spks+65))
                speakers[chr(spks+65)] = embedding


    return temp_start, temp_end, temp_speaker

@st.cache(suppress_st_warning=True,show_spinner=False)
def diarization_v3(audio,sr):
    diarization = diarization_inference(audio,sr)
    temp_start = []
    temp_end = []
    temp_speaker = []
    for elem in diarization:
        temp_start.append(elem["start"])
        temp_end.append(elem["stop"])
        temp_speaker.append(chr(int(elem["label"].split("_",1)[1])+65))
    temp_duration = [x1 - x2 for (x1, x2) in zip(temp_end, temp_start)]
    temp_asr_batches = asr_batches(temp_speaker,temp_duration)
    return pd.DataFrame(list(zip(temp_start,temp_end,temp_speaker,temp_duration,temp_asr_batches)),columns=["Start","End","Speaker","Duration","ASR Batch"])

def diarization_inference(audio,sr):
    split_val = 45
    split_threshold = 10
    if (len(audio)/sr)>(split_threshold+split_threshold):
        no_splits = calc_splits(len(audio)/sr,split_val,split_threshold)
        non_silent_interval = librosa.effects.split(audio, top_db=10, hop_length=1000)
        temp_dia = []
        mid_point = 0
        if no_splits < len(non_silent_interval):
            for i in range(no_splits):
                temp = int((len(non_silent_interval)/no_splits)*(i+1))-1
                if i < (no_splits-1):
                    dia_batch = dia_inf(audio[mid_point:int((non_silent_interval[temp][0]+non_silent_interval[temp-1][1])/2)],sr)
                    for elem in dia_batch:
                        elem["start"] += (mid_point/sr)
                        elem["stop"] += (mid_point/sr)
                    temp_dia.extend(dia_batch)
                    mid_point = int((non_silent_interval[temp][0]+non_silent_interval[temp-1][1])/2)
                else:
                    dia_batch = dia_inf(audio[mid_point:],sr)
                    for elem in dia_batch:
                        elem["start"] += (mid_point/sr)
                        elem["stop"] += (mid_point/sr)
                    temp_dia.extend(dia_batch)
            ##arrange temp dia
            return temp_dia  
        else: 
            return dia_inf(audio,sr)    
    else:
        return dia_inf(audio,sr)

def pyannote_query(filename):
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/pyannote/speaker-diarization"
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    temp = json.loads(response.content.decode("utf-8"))
    if "error" in temp:
        sys.stdout.write("Waiting for pyannote...\n")
        time.sleep(5)
        return pyannote_query(filename)
    return temp

def dia_inf(audio,sr):
    temp_dir = "C:\\Users\\mikea\\OneDrive\\Documents\\University\\Third Year\\Thesis\\Code\\audio\\temp.wav"
    write(temp_dir,sr,audio)
    return ast.literal_eval(pyannote_query(temp_dir)["text"])

def diarization_v2(audio):
    seg_output = seg_pipeline(audio)
    temp_start, temp_end, temp_speaker = seg_to_dia_v2(audio,seg_output)
    temp_duration = [x1 - x2 for (x1, x2) in zip(temp_end, temp_start)]
    temp_asr_batches = asr_batches(temp_speaker,temp_duration)
    df = pd.DataFrame(list(zip(temp_start,temp_end,temp_speaker,temp_duration,temp_asr_batches)),columns=["Start","End","Speaker","Duration","ASR Batch"])
    #st.dataframe(df)
    return df

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

#using HF Inference API
@st.cache(suppress_st_warning=True,show_spinner=False,allow_output_mutation=True)
def generate_transcript_dia_batches_v3(audio, sr, dia_df):
    
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
            #temp_transcript = enhance_model.add_punctuation_capitalization([temp_transcript])[0]
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
        
        temp_transcript += asr_inf(temp_audio,sr)
        t_md[-1].markdown("%s..." % temp_transcript)
        temp_speaker = temp_df["Speaker"].iloc[0]

    #temp_transcript = enhance_model.add_punctuation_capitalization([temp_transcript])[0]
    t_md[-1].markdown("%s" % temp_transcript)
    transcripts.append(temp_transcript)
    speakers.append(temp_speaker)
    for s, t in zip(speaker_md,t_md):
        s.empty()
        t.empty()

    return pd.DataFrame(list(zip(speakers, transcripts)),columns=["Speaker","Transcript"])


@st.cache(suppress_st_warning=True,show_spinner=False)
def generate_transcript_dia_batches_v2(audio, sr, dia_df):
    
    processor, model = load_W2V2()
    #checker = neuspell.BertChecker()
    #checker.from_pretrained()
    #enhance_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")
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
            #temp_transcript = enhance_model.add_punctuation_capitalization([temp_transcript])[0]
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

    #temp_transcript = enhance_model.add_punctuation_capitalization([temp_transcript])[0]
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

def w2v2_query(filename):
    API_TOKEN = "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"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h-lv60-self"
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    temp = json.loads(response.content.decode("utf-8"))
    if "error" in temp:
        sys.stdout.write("Waiting for Wav2Vec 2.0...\n")
        time.sleep(5)
        return w2v2_query(filename)
    else:
        return temp

def asr_transcript_inf(audio,sr):
    temp_dir = "C:\\Users\\mikea\\OneDrive\\Documents\\University\\Third Year\\Thesis\\Code\\audio\\temp.wav"
    write(temp_dir,sr,audio)
    return w2v2_query(temp_dir)["text"].lower()
    
def asr_inf(audio, sr):
    
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
                    temp_transcript += asr_transcript_inf(audio[mid_point:int((non_silent_interval[temp][0]+non_silent_interval[temp-1][1])/2)],sr)
                    temp_transcript += " "
                    mid_point = int((non_silent_interval[temp][0]+non_silent_interval[temp-1][1])/2)
                else:
                    temp_transcript += asr_transcript_inf(audio[mid_point:],sr)
            return temp_transcript   
        else: 
            return asr_transcript_inf(audio,sr)    
    else:
        return asr_transcript_inf(audio,sr)


def punct_query(payload):
    API_TOKEN = "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"
    API_URL = "https://api-inference.huggingface.co/models/SJ-Ray/Re-Punctuate"
    headers = {"Authorization": "Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def long_punct(text):
    tokenised = text.split()
    n = 50
    chunks = [' '.join(tokenised[i:i+n]) for i in range(0, len(tokenised), n)]
    sys.stdout.write("Text chunked into"+str(len(chunks))+"\n")
    st.write(chunks)
    if len(chunks[-1].split())<20:
        chunks[-2] = ' '.join(chunks[-2:])
        chunks.pop()
    long_transcript=""
    for chunk in chunks:
        long_transcript+=punct_query({
            "inputs": chunk,
            "options": {"wait_for_model": True}
        })[0]["generated_text"] + " "
        sys.stdout.write("chunk ready")
        

    sys.stdout.write("Long text punctuated\n")
    return long_transcript

@st.cache(suppress_st_warning=True,show_spinner=False,allow_output_mutation=True)
def optimise_transcript(transcript_df):
    transcript_df = transcript_df[transcript_df["Transcript"] != ""]
    
    raw_transcripts = transcript_df["Transcript"].tolist()
    ##checker = neuspell.BertChecker()
    #checker.from_pretrained()
    #corrected_transcripts = [checker.correct(str(sent)) for sent in raw_transcripts]
    #sys.stdout.write("Spell check complete")

    
    MAX_LEN = 100
    long_transcripts = {}
    for i, elem in enumerate(raw_transcripts):
        if len(elem.split())>MAX_LEN:
            sys.stdout.write("Long text detected\n")
            long_transcripts[i] = long_punct(elem)
            elem = "pass"
    
    output_punct = punct_query({
        "inputs": raw_transcripts,
        "options": {"wait_for_model": True}
    })
    sys.stdout.write("All text punctuated")
    enhanced_transcripts = []
    for elem in output_punct:
        st.write(elem)
        enhanced_transcripts.append(elem["generated_text"])
    for k,v in long_transcripts.items():
        enhanced_transcripts[k] = v
    
    transcript_df["Transcript"] = enhanced_transcripts
    return transcript_df


@st.cache(suppress_st_warning=True,show_spinner=False)
def gen_summary(transcript_df):
    time.sleep(5)


    return "This is a summary"
