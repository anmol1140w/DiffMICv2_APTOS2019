"""Evaluate calibration, selective risk and train-fitted Mahalanobis distances from saved outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from aptos_research.features import FeatureStore
from aptos_research.metrics import classification_metrics
from aptos_research.trust import fit_mahalanobis,mahalanobis,risk_coverage
from aptos_research.registry import record

def reliability_plot(labels, probabilities, path, bins=10):
 import matplotlib.pyplot as plt
 confidence=[max(row) for row in probabilities]; correct=[max(range(len(row)),key=lambda j:row[j])==int(y) for y,row in zip(labels,probabilities)]
 xs=[];ys=[]
 for b in range(bins):
  indices=[i for i,c in enumerate(confidence) if b/bins<=c<(b+1)/bins or (b==bins-1 and c==1)]
  if indices: xs.append(sum(confidence[i] for i in indices)/len(indices));ys.append(sum(correct[i] for i in indices)/len(indices))
 fig,ax=plt.subplots(figsize=(5,5));ax.plot([0,1],[0,1],'--',color='gray');ax.plot(xs,ys,'o-');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Accuracy',title='Reliability diagram');fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--results',required=True);p.add_argument('--model',default='ordered_esn_k16');p.add_argument('--output',default='outputs/experiments/trust.json');a=p.parse_args()
 experiment=json.loads(Path(a.results).read_text());store=FeatureStore(a.cache);tr=store.load_split('train');te=store.load_split('test')
 train_features=np.stack([x['states'][-1] for x in tr]);test_features=np.stack([x['states'][-1] for x in te]);stats=fit_mahalanobis(train_features)
 prediction=experiment['predictions'][a.model];labels=experiment['test']['labels'];probs=prediction['test_probabilities']
 result={'model':a.model,'classification':classification_metrics(labels,probs),'test_mahalanobis_distance':mahalanobis(stats,test_features).tolist(),'selective_risk_coverage':risk_coverage(labels,probs),'protocol':{'mahalanobis_fit':'training endpoint representations only','ece_bins':10,'risk_coverage':'descending maximum class confidence'}}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 plot_path=str(Path(a.output).with_suffix('.reliability.png'));reliability_plot(labels,probs,plot_path)
 record('outputs/experiments/registry.jsonl','trust-evaluation','COMPLETED',{'model':a.model},{'result':str(Path(a.output).resolve()),'reliability_diagram':plot_path},result['classification']);print('saved',a.output)
if __name__=='__main__':main()
