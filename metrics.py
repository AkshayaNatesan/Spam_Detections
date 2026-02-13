from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_metrics(y_true,y_pred,threshold=0.5):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    y_pred_bin=(y_pred>threshold).astype(int)
    f1=f1_score(y_true,y_pred_bin)
    return f1

def plot_confusion(y_true,y_pred,threshold=0.5):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    y_pred_bin=(y_pred>threshold).astype(int)
    cm=confusion_matrix(y_true,y_pred_bin)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()