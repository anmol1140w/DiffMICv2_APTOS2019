"""Training-only Mahalanobis OOD scores and selective risk/coverage."""
def fit_mahalanobis(train_features, ridge=1e-4):
 import numpy as np
 x=np.asarray(train_features,dtype=float)
 if x.ndim!=2 or len(x)<2: raise ValueError('training features must be [N,D], N>=2')
 mean=x.mean(axis=0); cov=np.cov(x,rowvar=False)
 cov=np.atleast_2d(cov)+ridge*np.eye(x.shape[1])
 return {'mean':mean,'precision':np.linalg.pinv(cov),'ridge':ridge}
def mahalanobis(model, features):
 import numpy as np
 d=np.asarray(features,dtype=float)-model['mean'];return np.sqrt(np.maximum(0,np.einsum('bi,ij,bj->b',d,model['precision'],d)))
def risk_coverage(labels,probabilities):
 import numpy as np
 y=np.asarray(labels);p=np.asarray(probabilities);conf=p.max(axis=1);pred=p.argmax(axis=1);order=np.argsort(-conf);err=(pred!=y)[order]
 return [{'coverage':float(i/len(y)),'risk':float(err[:i].mean()) if i else 0.0,'threshold':float(conf[order[i-1]]) if i else 1.0} for i in range(1,len(y)+1)]
def corrupt_brightness(images,offset):
 import numpy as np
 return np.clip(np.asarray(images,dtype=float)+float(offset),0,1)
