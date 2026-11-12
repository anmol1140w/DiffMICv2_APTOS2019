"""Metric definitions and calibration helpers for five-class ordinal APTOS."""
import math

def _matrix(labels, predictions, classes=5):
    cm=[[0]*classes for _ in range(classes)]
    for y,p in zip(labels,predictions):
        y,p=int(y),int(p)
        if not 0<=y<classes or not 0<=p<classes: raise ValueError('labels/predictions must be grades 0..4')
        cm[y][p]+=1
    return cm

def classification_metrics(labels, probabilities):
    if len(labels)!=len(probabilities) or not labels: raise ValueError('non-empty aligned labels/probabilities required')
    k=len(probabilities[0]); pred=[max(range(k),key=lambda j:row[j]) for row in probabilities]
    cm=_matrix(labels,pred,k); n=len(labels)
    f1=[]
    for c in range(k):
        tp=cm[c][c]; fp=sum(cm[r][c] for r in range(k) if r!=c); fn=sum(cm[c][j] for j in range(k) if j!=c)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    total=sum(cm[i][j] for i in range(k) for j in range(k)); observed=sum(cm[i][j]*(i-j)**2 for i in range(k) for j in range(k))/max(1,total)
    rows=[sum(row) for row in cm]; cols=[sum(cm[i][j] for i in range(k)) for j in range(k)]
    expected=sum(rows[i]*cols[j]*(i-j)**2 for i in range(k) for j in range(k))/max(1,total*total)
    brier=sum(sum((float(row[j])-(j==int(y)))**2 for j in range(k)) for y,row in zip(labels,probabilities))/n
    confidence=[max(row) for row in probabilities]; correct=[p==int(y) for p,y in zip(pred,labels)]
    ece=0.0
    for b in range(10):
        ix=[i for i,c in enumerate(confidence) if (b/10<=c<(b+1)/10) or (b==9 and c==1)]
        if ix: ece+=len(ix)/n*abs(sum(correct[i] for i in ix)/len(ix)-sum(confidence[i] for i in ix)/len(ix))
    aucs=[]
    for c in range(k):
        positive=[float(row[c]) for y,row in zip(labels,probabilities) if int(y)==c]
        negative=[float(row[c]) for y,row in zip(labels,probabilities) if int(y)!=c]
        if positive and negative:
            aucs.append(sum((1.0 if p>q else .5 if p==q else 0.0) for p in positive for q in negative)/(len(positive)*len(negative)))
    return {'accuracy':sum(pred[i]==int(labels[i]) for i in range(n))/n,'macro_f1':sum(f1)/k,'per_class_f1':f1,'quadratic_weighted_kappa':1-observed/expected if expected else 0.0,'ordinal_mae':ordinal_mae(labels,pred),'macro_auroc_ovr':sum(aucs)/len(aucs) if aucs else None,'brier_score':brier,'ece_10_bins':ece,'confusion_matrix':cm,'predictions':pred}

def ordinal_mae(labels, predictions): return sum(abs(int(a)-int(b)) for a,b in zip(labels,predictions))/len(labels)
