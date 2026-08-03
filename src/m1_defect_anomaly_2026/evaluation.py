import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix




def evaluate_scores(y_true, scores, threshold,method="unnamed"):
    y_pred=(scores>=threshold).astype(int)
    precision,recall, f1, _= precision_recall_fscore_support(y_true,y_pred, average="binary",zero_division=0)


    if len(np.unique(y_true))<2:
        roc=float("nan")
    else:
        roc=roc_auc_score(y_true,scores)

    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()

    return { "method": method,
             "precision":float(precision),
             "recall":float(recall),
             "f1":float(f1),
             "roc_auc":float(roc),
             "threshold":float(threshold),
             "tp":int(tp),
             "fp":int(fp),
             "fn":int(fn),
             "tn":int(tn),
             "n_beats":int(len(y_true)),
             "anomaly_rate":float(y_true.mean())}



