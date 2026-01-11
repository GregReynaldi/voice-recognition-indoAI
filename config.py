from datasets import Dataset
import torch
import re

def get_feature(minds_14 : Dataset , index : int) :  
    data1 = minds_14[index]
    audio = data1["audio"]["array"]
    sr = data1["audio"]["sampling_rate"]
    label = data1["transcription"]
    translate = data1["english_transcription"]
    return audio, sr, label, translate

def generateAudio2Text(minds_14, index,processor,model) : 
    audio, sr, label, translate = get_feature(minds_14, index)
    audio_encoded = processor(audio, sampling_rate = sr, return_tensors = "pt")
    with torch.no_grad() : 
        hasil = model.generate(**audio_encoded)
    answer = processor.batch_decode(hasil)
    return answer, label, translate

def cleaning_text(sentence) : 
    sentence = sentence.lower().strip()
    sentence = re.sub(string = sentence, pattern = r'[?,.!\']', repl = " ")
    sentence = re.sub(string = sentence, pattern = r"\s+", repl = " ")
    return sentence