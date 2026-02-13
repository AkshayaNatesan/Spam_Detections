import pandas as pd
import torch
import torch.nn as nn
import pickle
from models.data_loader import get_dataloader
from models.spam_detector import SpamDetector

df=pd.read_csv("data/spam.csv",encoding='latin1')
df=df[['v1','v2']].dropna()
texts=df['v2'].tolist()
labels=[1 if x=='spam' else 0 for x in df['v1']]
train_size=int(0.8*len(texts))
val_size=int(0.1*len(texts))
test_size=len(texts)-train_size-val_size
train_texts,val_texts,test_texts=texts[:train_size],texts[train_size:train_size+val_size],texts[train_size+val_size:]
train_labels,val_labels,test_labels=labels[:train_size],labels[train_size:train_size+val_size],labels[train_size+val_size:]
train_loader,vocab=get_dataloader(train_texts,train_labels,batch_size=32)
with open("vocab.pkl","wb") as f:
    pickle.dump(vocab,f)
val_loader,_=get_dataloader(val_texts,val_labels,batch_size=32,vocab=vocab)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=SpamDetector(len(vocab)).to(device)
spam_count=sum(train_labels)
ham_count=len(train_labels)-spam_count
pos_weight=torch.tensor([ham_count/spam_count]).to(device)
criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs=15
for epoch in range(epochs):
    model.train()
    total_loss=0
    for seqs,labels_batch in train_loader:
        seqs=seqs.long().to(device)
        labels_batch=labels_batch.to(device).view(-1)
        optimizer.zero_grad()
        outputs=model(seqs)
        loss=criterion(outputs,labels_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}/{epochs},Loss:{total_loss/len(train_loader):.4f}")
torch.save(model.state_dict(),"spam_detector.pth")
print("Training complete! Model saved.")