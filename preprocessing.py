import re
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch
def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+","",text)
    text=re.sub(r"\d+","",text)
    text=re.sub(r"[^\w\s]","",text)
    text=text.strip()
    return text
def encode_labels(labels):
    le=LabelEncoder()
    return torch.tensor(le.fit_transform(labels),dtype=torch.float32)
def pad_sequences(seq_list,padding_value=0):
    return pad_sequence(seq_list,batch_first=True,padding_value=padding_value)
def pad_sequences(seq_list,padding_value=0):
    return pad_sequences(seq_list,padding_values=0)