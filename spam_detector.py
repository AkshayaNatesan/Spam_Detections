import torch
import torch.nn as nn

class SpamDetector(nn.Module):
    def __init__(self,vocab_size,embed_dim=50,hidden_dim=64):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_dim*2,1)
        self.fc=nn.Linear(hidden_dim,1)
    def forward(self,x):
        x=self.embedding(x)
        _,(h,_)=self.lstm(x)
        out=self.fc(h[-1])
        return out.squeeze()