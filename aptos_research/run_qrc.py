"""Gate-controlled QRC evaluation. Requires a PASS gate artifact."""
import argparse, json
from pathlib import Path
from aptos_research.features import FeatureStore
from aptos_research.gate import qrc_is_eligible
from aptos_research.models import qrc_features,fit_classifier,make_esn,esn_transform
from aptos_research.metrics import classification_metrics
from aptos_research.registry import record

def main():
 p=argparse.ArgumentParser();p.add_argument('--gate',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',default='outputs/experiments/qrc.json');p.add_argument('--registry',default='outputs/experiments/registry.jsonl');p.add_argument('--qubits',type=int,default=4);p.add_argument('--shots',type=int,default=0);p.add_argument('--seed',type=int,default=2026);a=p.parse_args()
 if not qrc_is_eligible(a.gate):
  result={'status':'SKIPPED_GATE_NOT_PASS','gate':json.loads(Path(a.gate).read_text()).get('decision')}
  Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');record(a.registry,'qrc','SKIPPED_GATE_NOT_PASS',{'gate':result['gate']},{'result':str(Path(a.output).resolve())},result);print(json.dumps(result));return 2
 import numpy as np
 store=FeatureStore(a.cache);tr=store.load_split('train');va=store.load_split('validation');te=store.load_split('test')
 def seq(rows): return np.stack([x['states'] for x in rows])
 xt,meta=qrc_features(seq(tr),a.qubits,a.seed,a.shots); xv,_=qrc_features(seq(va),a.qubits,a.seed+1,a.shots); xe,_=qrc_features(seq(te),a.qubits,a.seed+2,a.shots)
 model=fit_classifier(xt,[x['label'] for x in tr],a.seed).fit(xt,[x['label'] for x in tr]); valp=model.predict_proba(xv); testp=model.predict_proba(xe)
 ytr=[x['label'] for x in tr]; yv=[x['label'] for x in va]; yt=[x['label'] for x in te]
 # Same compact trajectory inputs for a classical reservoir reference.
 xtr=seq(tr);xv0=seq(va);xte0=seq(te); reservoir=make_esn(xtr.shape[-1],64,a.seed)
 er=fit_classifier(esn_transform(xtr,reservoir),ytr,a.seed).fit(esn_transform(xtr,reservoir),ytr)
 evp=er.predict_proba(esn_transform(xv0,reservoir));etp=er.predict_proba(esn_transform(xte0,reservoir))
 result={'status':'COMPLETED_CONDITIONAL','gate_decision':'PASS','config':meta,'validation':{'qrc':classification_metrics(yv,valp.tolist()),'esn':classification_metrics(yv,evp.tolist())},'test':{'qrc':classification_metrics(yt,testp.tolist()),'esn':classification_metrics(yt,etp.tolist())},'resource_accounting':{'qrc_readout_parameters':int(xt.shape[1]*5+5),'esn_reservoir_parameters':int(reservoir['input'].size+reservoir['reservoir'].size),'esn_readout_parameters':int(er[-1].coef_.size+er[-1].intercept_.size),'state_dimension':int(xtr.shape[-1]),'trajectory_length':int(xtr.shape[1]),'quantum_hardware_execution':False},'note':'Resource and predictive comparisons are simulator-only; test results are descriptive and do not change the precomputed gate.'}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');record(a.registry,'qrc','COMPLETED_CONDITIONAL',meta,{'result':str(Path(a.output).resolve())},result['test']);print(json.dumps(result,indent=2))
if __name__=='__main__':raise SystemExit(main())
