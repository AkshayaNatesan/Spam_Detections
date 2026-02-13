import pandas as pd
import torch
import pickle
from models.spam_detector import SpamDetector
from models.data_loader import get_dataloader
from utils.metrics import compute_metrics,plot_confusion

df=pd.read_csv("data/spam.csv",encoding='latin1')
df=df[['v1','v2']].dropna()
texts=df['v2'].tolist()
labels=[1 if x=='spam' else 0 for x in df['v1']]
with open("vocab.pkl","rb") as f:
    vocab=pickle.load(f)
test_loader, _=get_dataloader(texts,labels,batch_size=32,vocab=vocab)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=SpamDetector(len(vocab)).to(device)
model.load_state_dict(torch.load("spam_detector.pth"))
model.eval()
all_preds=[]
all_labels=[]
with torch.no_grad():
    for seqs,lbls in test_loader:
        seqs=seqs.long().to(device)
        outputs=model(seqs).cpu().numpy()
        all_preds.extend(outputs)
        all_labels.extend(lbls.numpy())
f1=compute_metrics(all_labels,all_preds)
print(f"Spam Class F1 Score:{f1*100:.2f}%")
plot_confusion(all_labels,all_preds)
print("Sample predictions:",all_preds[:20]) 