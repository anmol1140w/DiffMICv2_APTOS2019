"""Instrument real DiffMIC SR3Sampler runs through a project adapter.

Adapter contract: factory() -> (sampler, iterable[dict]) where each batch dict has
ids, labels, splits, x_batch, noisy_y, conditions. stochastic initial states must be
seeded in the adapter; the extractor records the seed/configuration in SQLite.
"""
import argparse, hashlib, importlib, json, random
from pathlib import Path
from aptos_research.features import FeatureStore
from aptos_research.registry import record

def _resolve(spec):
 module,name=spec.split(':',1); return getattr(importlib.import_module(module),name)
def extract(factory_spec, cache_path, checkpoint, seed=2026, max_samples=None, inference_steps=16, pooling='flatten'):
 import numpy as np, torch
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 sampler,batches=_resolve(factory_spec)()
 scheduler=sampler.scheduler
 if inference_steps: scheduler.set_timesteps(inference_steps)
 ts=[int(t) for t in scheduler.timesteps]
 if not ts: raise ValueError('sampler scheduler has no timesteps; call set_timesteps in adapter')
 if pooling not in ('flatten','gap'): raise ValueError('pooling must be flatten or gap')
 config={'sampler':'SR3Sampler','state':'post-step class-score diffusion grid (not an image latent)','full_reverse_steps':len(ts),'state_selection':'uniform indices over reverse-loop order','projection':'flatten spatial class-score grid' if pooling=='flatten' else 'global average pool spatial class-score grid','input_state_shape':'recorded from callback','seed':seed}
 k=min(16,len(ts)); selected_indices=[len(ts)-1] if k==1 else np.linspace(0,len(ts)-1,k,dtype=int).tolist(); wanted={ts[i]:i for i in selected_indices}
 ckpt_hash=getattr(sampler,'checkpoint_id',None)
 if ckpt_hash is None and checkpoint and checkpoint!='unavailable':
  digest=hashlib.sha256()
  with open(checkpoint,'rb') as stream:
   for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
  ckpt_hash=digest.hexdigest()
 ckpt_hash=ckpt_hash or 'unavailable'
 store=FeatureStore(cache_path); seen=0
 for batch in batches:
  all_ids=batch['ids'];all_labels=batch['labels'];all_splits=batch['splits']
  keep=[i for i,(sid,label,split) in enumerate(zip(all_ids,all_labels,all_splits)) if not store.contains(sid,split,int(label),ckpt_hash,config,seed)]
  if not keep: continue
  indices=torch.as_tensor(keep,dtype=torch.long,device=batch['x_batch'].device)
  def take(value):
   if torch.is_tensor(value) and value.ndim>0 and value.shape[0]==len(all_ids): return value.index_select(0,indices.to(value.device))
   if isinstance(value,list): return [take(v) for v in value]
   if isinstance(value,tuple): return tuple(take(v) for v in value)
   return value
  ids=[all_ids[i] for i in keep];labels=[all_labels[i] for i in keep];splits=[all_splits[i] for i in keep]
  x_batch=take(batch['x_batch']);noisy_y=take(batch['noisy_y']);conditions=take(batch['conditions']);captures={i:[] for i in range(len(ids))}
  def callback(timestep,state_index,state):
   if timestep not in wanted: return
   if state.ndim!=4 or state.shape[0]!=len(ids) or state.shape[1]!=5: raise RuntimeError('Expected five-class diffusion state [B,5,H,W], got %s' % (tuple(state.shape),))
   config['input_state_shape']=list(state.shape[1:])
   projected=state.detach().float().flatten(1) if pooling=='flatten' else state.detach().float().mean(dim=(-2,-1))
   pooled=projected.cpu().numpy()
   for j,v in enumerate(pooled): captures[j].append((wanted[timestep],timestep,v))
  sampler.sample_high_res(x_batch,noisy_y,conditions=conditions,trajectory_callback=callback,selected_timesteps=wanted.keys())
  for j,sid in enumerate(ids):
   values=sorted(captures[j],key=lambda z:z[0])
   if len(values)!=k: raise RuntimeError('Captured %d/%d selected states for %s' % (len(values),k,sid))
   store.add_sample(sid,splits[j],int(labels[j]),np.stack([x[2] for x in values]),[x[1] for x in values],ckpt_hash,config,seed); seen+=1
   if max_samples and seen>=max_samples: store.close(); return seen
 store.close(); return seen

def main():
 p=argparse.ArgumentParser();p.add_argument('--factory',required=True,help='Python module:function adapter for real DiffMIC model/checkpoint and APTOS batches');p.add_argument('--cache',default='outputs/trajectories.sqlite');p.add_argument('--checkpoint',required=True);p.add_argument('--seed',type=int,default=2026);p.add_argument('--max-samples',type=int);p.add_argument('--inference-steps',type=int,default=16);p.add_argument('--pooling',choices=['flatten','gap'],default='flatten');a=p.parse_args()
 count=extract(a.factory,a.cache,a.checkpoint,a.seed,a.max_samples,a.inference_steps,a.pooling)
 record('outputs/experiments/registry.jsonl','trajectory-extraction','PARTIAL_BOUNDED' if a.max_samples else 'COMPLETED',{'factory':a.factory,'seed':a.seed,'inference_steps':a.inference_steps,'pooling':a.pooling,'max_samples':a.max_samples},{'feature_cache':str(Path(a.cache).resolve())},{'new_samples':count})
 print(json.dumps({'new_samples':count}))
if __name__=='__main__':main()
