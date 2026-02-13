import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import clean_text
from collections import Counter
class SMSDataset(Dataset):
    def __init__(self,texts,labels,vocab=None,max_len=50):
        self.texts=[clean_text(t).split() for t in texts]
        self.labels=torch.tensor(labels,dtype=torch.float32)
        self.max_len=max_len
        if vocab is None:
            self.vocab={"<PAD>":0,"<UNK>":1}
            counter=Counter(word for text in self.texts for word in text)
            for i,word in enumerate(counter.keys(),start=2):
                self.vocab[word]=i
        else:
            self.vocab=vocab
    def text_to_seq(self,text):
        seq=[self.vocab.get(word, self.vocab["<UNK>"]) for word in text]
        if len(seq)<self.max_len:
            seq+=[0]*(self.max_len-len(seq))
        else:
            seq=seq[:self.max_len]
        return torch.tensor(seq)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self,idx):
        seq=self.text_to_seq(self.texts[idx])
        label=self.labels[idx]
        return seq,label
def get_dataloader(texts,labels,batch_size=32,shuffle=True,vocab=None):
    dataset=SMSDataset(texts,labels,vocab=vocab)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle),dataset.vocab