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






# find a threshold with the best F1 while using the training set
def select_threshold(y_true,scores,n_candidates=200):

    candidates=np.percentile(scores,np.linspace(0.5,99.9,n_candidates))
    candidates=np.unique(candidates)
    best_threshold=candidates[0]
    best_f1=-1.0

    for threshold in candidates:
        y_pred=(scores>=threshold).astype(int)
        _,_,f1,_=precision_recall_fscore_support(y_true,y_pred,average="binary",zero_division=0)
        if f1> best_f1:
            best_f1=f1
            best_threshold=threshold



    return float(best_threshold), float(best_f1)






# create a table that saves the results 

def build_table(results):
    df=pd.DataFrame(results)
    columns=["method","precision","recall","f1","roc_auc","threshold","tp","fp","fn",
             "tn","n_beats","anomaly_rate"]
    return df[columns].sort_values("f1",ascending=False).reset_index(drop=True)



def save_table(df,csv_path,md_path=None):
    df.to_csv(csv_path,index=False)
    if md_path:
        show=df[["method","precision","recall","f1","roc_auc"]].copy()
        for c in ["precision","recall","f1","roc_auc"]:
            show[c]=show[c].map(lambda v:f"{v:.3f}")
        with open(md_path,"w") as f:
            f.write("# Method comparison (test set=DS2)\n\n")
            f.write(show.to_markdown(index=False))
            f.write("\n")





