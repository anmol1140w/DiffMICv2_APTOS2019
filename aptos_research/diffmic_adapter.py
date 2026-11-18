"""Concrete adapter for the repository's ConditionalModel + DCG + SR3Sampler."""
import os, yaml
from pathlib import Path

def _state(payload, prefix=None):
 if isinstance(payload,(tuple,list)) and payload and isinstance(payload[0],dict): payload=payload[0]
 if not isinstance(payload,dict): raise ValueError('Checkpoint must contain a state dictionary')
 if 'state_dict' in payload and isinstance(payload['state_dict'],dict): payload=payload['state_dict']
 if prefix:
  stripped={k[len(prefix):]:v for k,v in payload.items() if k.startswith(prefix)}
  if stripped: return stripped
 return payload

def factory():
 import torch
 from easydict import EasyDict
 from torch.utils.data import DataLoader
 from aptos_research.data import load_split_manifest,AptosImageDataset
 from aptos_research.checkpoint import load_compatible
 from model import ConditionalModel
 from pretraining.dcg import DCG
 from pipeline import SR3Sampler,SR3scheduler
 cfg_path = os.environ.get('DIFFMIC_CONFIG', 'configs/aptos.yml')
 config = yaml.safe_load(Path(cfg_path).read_text())
 root=os.environ.get('APTOS_ROOT',config.get('data',{}).get('dataroot','/home/pratyum/diffmic/data/aptos'))
 manifest=os.environ.get('APTOS_SPLITS',config.get('data',{}).get('splits_manifest','outputs/aptos_splits.json'))
 ckpt=os.environ.get('DIFFMIC_CHECKPOINT','/home/pratyum/diffmic/checkpoints/v2/ckpts/aptos-epoch117-acc-0.8186-f1-0.6840.ckpt')
 aux_ckpt=os.environ.get('DIFFMIC_AUX_CHECKPOINT',config.get('data',{}).get('aux_checkpoint','/home/pratyum/diffmic/code/DiffMICv2/pretraining/ckpt/aptos_aux_model.pth'))
 if not root or not Path(root).is_dir(): raise RuntimeError('Set APTOS_ROOT to the real APTOS directory: %s' % root)
 if not ckpt or not Path(ckpt).is_file(): raise RuntimeError('Set DIFFMIC_CHECKPOINT to an existing trained ConditionalModel checkpoint: %s' % ckpt)
 if not aux_ckpt or not Path(aux_ckpt).is_file(): raise RuntimeError('Set DIFFMIC_AUX_CHECKPOINT to an existing trained DCG checkpoint: %s' % aux_ckpt)
 splits=load_split_manifest(manifest)
 config['data']['dataroot']=root; cfg=EasyDict(config)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 denoiser=ConditionalModel(cfg,guidance=True).to(device)
 aux=DCG(cfg).to(device)
 try:
  den_payload=torch.load(ckpt,map_location='cpu',weights_only=False)
 except TypeError:
  den_payload=torch.load(ckpt,map_location='cpu')
 try:
  aux_payload=torch.load(aux_ckpt,map_location='cpu',weights_only=False)
 except TypeError:
  aux_payload=torch.load(aux_ckpt,map_location='cpu')
 dr=load_compatible(denoiser,_state(den_payload,'model.'),strict_fraction=.95)
 ar=load_compatible(aux,_state(aux_payload,'aux_model.'),strict_fraction=.95)
 print('ConditionalModel checkpoint compatibility:',dr['loaded_fraction'],len(dr['mismatched_shapes']))
 print('DCG checkpoint compatibility:',ar['loaded_fraction'],len(ar['mismatched_shapes']))
 denoiser.eval();aux.eval()
 scheduler=SR3scheduler(num_train_timesteps=1000,beta_start=.0001,beta_end=.02,beta_schedule='linear',diff_chns=5)
 scheduler.set_timesteps(16)
 sampler=SR3Sampler(denoiser,scheduler,eta=0.0)
 import hashlib
 def file_hash(path):
  h=hashlib.sha256()
  with open(path,'rb') as stream:
   for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
  return h.hexdigest()
 sampler.checkpoint_id='denoiser:'+file_hash(ckpt)+';dcg:'+file_hash(aux_ckpt)
 def batches():
  for split in ('train','validation','test'):
   records=splits['splits'][split]
   ds=AptosImageDataset(records,split,size=224,training=False)
   loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)
   for image,label,ids,split_names in loader:
    image=image.to(device);label=label.to(device)
    with torch.no_grad():
     y,y_global,y_local,patches,attns,attn_map=aux(image)
     b,nc=y.shape; n=attns.shape[1]
     distance=torch.abs(torch.arange(n,device=device)[:,None]-torch.arange(n,device=device)[None,:])
     wg=1-distance/(n-1);wl=distance/(n-1)
     cond=wl[None,None]*y_local[:,:,None,None]+wg[None,None]*y_global[:,:,None,None]
     diag=torch.arange(n,device=device)
     for j in range(b):
      for c in range(nc):
       cond[j,c,diag,diag]=y_global[j,c];cond[j,c,n-1,0]=y_local[j,c];cond[j,c,0,n-1]=y_local[j,c]
     attn=attns.unsqueeze(-1); attn=(attn*attn.transpose(1,2)).unsqueeze(1)
     rg=torch.rand_like(y_global); rl=torch.rand_like(y_local)
     noisy=wl[None,None]*rl[:,:,None,None]+wg[None,None]*rg[:,:,None,None]
     for j in range(b):
      for c in range(nc):
       noisy[j,c,diag,diag]=rg[j,c]; noisy[j,c,n-1,0]=rl[j,c]; noisy[j,c,0,n-1]=rl[j,c]
    yield {'ids':list(ids),'labels':label.cpu().tolist(),'splits':list(split_names),'x_batch':image,'noisy_y':noisy,'conditions':[cond,patches,attn]}
 return sampler,batches()
