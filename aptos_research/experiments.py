"""Static and classical trajectory controls on one shared feature cache."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from aptos_research.features import FeatureStore
from aptos_research.metrics import classification_metrics,ordinal_mae
from aptos_research.models import pool_trajectory,make_esn,esn_transform,fit_classifier
from aptos_research.registry import record

def selected(rows,k):
 out=[]
 for r in rows:
  x=r['states']; ix=np.array([len(x)-1]) if k==1 else np.linspace(0,len(x)-1,min(k,len(x)),dtype=int); out.append(x[ix])
 return np.stack(out)
def fit_eval(train,valid,test,seed,pca_dim=0,output=None,include_gru=False):
 if pca_dim:
  from sklearn.decomposition import PCA
  from joblib import dump
  pca=PCA(n_components=pca_dim,random_state=seed).fit(np.concatenate([x['states'] for x in train],axis=0))
  for rows in (train,valid,test):
   for row in rows: row['states']=pca.transform(row['states']).astype(np.float32)
  if output:
   Path(output).parent.mkdir(parents=True,exist_ok=True);dump(pca,str(output)+'.pca.joblib')
 ytr=np.array([x['label'] for x in train]); yv=np.array([x['label'] for x in valid]); yt=np.array([x['label'] for x in test]); result={'validation':{'labels':yv.tolist(),'ids':[x['id'] for x in valid]},'test':{'labels':yt.tolist(),'ids':[x['id'] for x in test]},'models':{},'predictions':{}}
 def evaluate(name,xtr,xv,xt):
  start=time.perf_counter(); model=fit_classifier(xtr,ytr,seed).fit(xtr,ytr); fit_s=time.perf_counter()-start
  vprob=model.predict_proba(xv); t0=time.perf_counter(); tprob=model.predict_proba(xt); latency=(time.perf_counter()-t0)/max(1,len(xt))
  vm=classification_metrics(yv,vprob.tolist()); tm=classification_metrics(yt,tprob.tolist());
  result['models'][name]={'parameters':int(model[-1].coef_.size+model[-1].intercept_.size),'fit_seconds':fit_s,'test_latency_seconds_per_sample':latency,'validation':{k:v for k,v in vm.items() if k!='predictions'},'test':{k:v for k,v in tm.items() if k!='predictions'}}
  if name.startswith('ordered_esn_'): result['validation']['ordered_predictions']=vm['predictions']
  result['validation'][name+'_predictions']=vm['predictions']; result['test'][name+'_predictions']=tm['predictions']
  result['predictions'][name]={'validation_probabilities':vprob.tolist(),'test_probabilities':tprob.tolist(),'validation_labels':yv.tolist(),'test_labels':yt.tolist()}
 k_candidates = [k for k in (1,4,8,10,16) if len(train[0]['states'])>=k]
 max_k = max(k_candidates) if k_candidates else len(train[0]['states'])
 for k in k_candidates:
  tr=selected(train,k); va=selected(valid,k); te=selected(test,k)
  for name,fn in [('ordered_mean',lambda x:x.mean(axis=1)),('ordered_last',lambda x:x[:,-1])]: evaluate('%s_k%d'%(name,k),fn(tr),fn(va),fn(te))
  evaluate('unordered_mean_k%d'%k,pool_trajectory(tr),pool_trajectory(va),pool_trajectory(te))
  # Fixed cyclic permutation, same states exactly; record permutation seed.
  perm=np.roll(np.arange(k),1); evaluate('shuffled_k%d'%k,tr[:,perm].mean(axis=1),va[:,perm].mean(axis=1),te[:,perm].mean(axis=1))
  esn=make_esn(tr.shape[-1],64,seed+k); evaluate('ordered_esn_k%d'%k,esn_transform(tr,esn),esn_transform(va,esn),esn_transform(te,esn)); evaluate('shuffled_esn_k%d'%k,esn_transform(tr[:,perm],esn),esn_transform(va[:,perm],esn),esn_transform(te[:,perm],esn))
  reservoir_params=esn['input'].size+esn['reservoir'].size
  for key in ('ordered_esn_k%d'%k,'shuffled_esn_k%d'%k): result['models'][key]['parameters']+=int(reservoir_params)
 # Validation-only best individual state; the held-out test is evaluated only after selection.
 best_idx=None; best_score=-1.0
 for state_idx in range(len(train[0]['states'])):
  evaluate('single_timestep_%d'%state_idx,np.stack([x['states'][state_idx] for x in train]),np.stack([x['states'][state_idx] for x in valid]),np.stack([x['states'][state_idx] for x in test]))
  score=result['models']['single_timestep_%d'%state_idx]['validation']['macro_f1']
  if score>best_score: best_score=score;best_idx=state_idx
 result['best_single_timestep']={'state_index':best_idx,'selection_metric':'validation macro-F1','validation_macro_f1':best_score,'test_metrics':result['models']['single_timestep_%d'%best_idx]['test']}
 # Gate consumes a pre-named matched set of relevant controls only.
 result['validation']['control_predictions']=[result['validation'][f'ordered_last_k{max_k}_predictions'],result['validation']['single_timestep_%d_predictions'%best_idx],result['validation'][f'unordered_mean_k{max_k}_predictions'],result['validation'][f'shuffled_esn_k{max_k}_predictions']]
 if include_gru:
  from aptos_research.train_gru import train_gru
  train_seq=selected(train,max_k); val_seq=selected(valid,max_k); test_seq=selected(test,max_k)
  mean=train_seq.mean(axis=(0,1),keepdims=True);std=train_seq.std(axis=(0,1),keepdims=True)+1e-6
  train_seq=(train_seq-mean)/std;val_seq=(val_seq-mean)/std;test_seq=(test_seq-mean)/std
  for name,tr_seq,va_seq,te_seq in [(f'ordered_gru_k{max_k}',train_seq,val_seq,test_seq),(f'shuffled_gru_k{max_k}',np.roll(train_seq,1,axis=1),np.roll(val_seq,1,axis=1),np.roll(test_seq,1,axis=1))]:
   start=time.perf_counter();model,vprob,tprob,param_count=train_gru(tr_seq,ytr,va_seq,yv,te_seq,seed=seed,hidden=64,epochs=80,patience=10); elapsed=time.perf_counter()-start
   vm=classification_metrics(yv,vprob.tolist());tm=classification_metrics(yt,tprob.tolist())
   result['models'][name]={'parameters':param_count,'fit_seconds':elapsed,'validation':{k:v for k,v in vm.items() if k!='predictions'},'test':{k:v for k,v in tm.items() if k!='predictions'}}
   result['validation'][name+'_predictions']=vm['predictions'];result['test'][name+'_predictions']=tm['predictions']
   result['predictions'][name]={'validation_probabilities':vprob.tolist(),'test_probabilities':tprob.tolist(),'validation_labels':yv.tolist(),'test_labels':yt.tolist()}
   if name==f'shuffled_gru_k{max_k}': result['validation']['control_predictions'].append(vm['predictions'])
 result['config']={'seed':seed,'classifier':'standardized multinomial logistic regression, max_iter=1000','pca_dim':pca_dim or None,'pca_fit_scope':'training states only','gru_included':include_gru,'esn':{'hidden_size':64,'connectivity':.1,'spectral_radius':.9,'leak':.5,'readout':'logistic regression'},'shuffle':'cyclic rotation by one; fixed, same states','best_single_timestep_selection':'validation macro-F1','test_use':'locked descriptive evaluation; validation controls selection'}
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',default='outputs/experiments/classical.json');p.add_argument('--seed',type=int,default=2026);p.add_argument('--pca-dim',type=int,default=0);p.add_argument('--gru',action='store_true',help='also run ordered and shuffled GRU; needs PyTorch');a=p.parse_args();s=FeatureStore(a.cache);tr=s.load_split('train');va=s.load_split('validation');te=s.load_split('test')
 if not tr or not va or not te: raise SystemExit('Feature cache must contain non-empty train, validation, and test splits')
 result=fit_eval(tr,va,te,a.seed,a.pca_dim,a.output,a.gru);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');record('outputs/experiments/registry.jsonl','classical-experiments','COMPLETED',result['config'],{'results':str(Path(a.output).resolve())},result.get('best_single_timestep',{}));print('saved',a.output)
if __name__=='__main__':main()
