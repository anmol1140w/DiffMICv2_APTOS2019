"""Plot class-score-derived probability trajectories for IDs in the feature cache."""
import argparse
from pathlib import Path
from aptos_research.features import FeatureStore
from aptos_research.explain import class_scores_to_probabilities,save_probability_trajectory
from aptos_research.registry import record

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--sample-id',required=True);p.add_argument('--split',choices=['train','validation','test'],default='test');p.add_argument('--output',default='outputs/experiments/xai');a=p.parse_args()
 rows=FeatureStore(a.cache).load_split(a.split);row=next((x for x in rows if x['id']==a.sample_id),None)
 if row is None: raise SystemExit('sample id not found in cache split: '+a.sample_id)
 shape=row['config'].get('input_state_shape')
 if not shape or len(shape)!=3: raise SystemExit('state cache lacks verified [classes,height,width] metadata')
 probs=class_scores_to_probabilities(row['states'],int(shape[0]),(int(shape[1]),int(shape[2])))
 target=Path(a.output);target.mkdir(parents=True,exist_ok=True);path=target/(a.sample_id+'_trajectory.png')
 save_probability_trajectory(row['timesteps'],probs,str(path),'DiffMIC class-score state softmax: '+a.sample_id)
 record('outputs/experiments/registry.jsonl','trajectory-explanation','COMPLETED',{'sample_id':a.sample_id,'split':a.split,'derived':'spatially averaged class-score grid followed by softmax'},{'plot':str(path.resolve())},{'grade':row['label']});print(path)
if __name__=='__main__':main()
