"""Small deterministic GRU sequence classifier for [N,K,D] feature arrays."""
def train_gru(train_x,train_y,val_x,val_y,test_x=None,seed=2026,hidden=64,epochs=100,patience=12,device='cpu'):
 import copy, numpy as np, torch
 from torch import nn
 torch.manual_seed(seed); np.random.seed(seed)
 class Net(nn.Module):
  def __init__(self,d,h):
   super().__init__();self.gru=nn.GRU(d,h,batch_first=True);self.head=nn.Linear(h,5)
  def forward(self,x):return self.head(self.gru(x)[0][:,-1])
 model=Net(train_x.shape[-1],hidden).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
 tx=torch.tensor(train_x,dtype=torch.float32,device=device);ty=torch.tensor(train_y,dtype=torch.long,device=device);vx=torch.tensor(val_x,dtype=torch.float32,device=device);vy=torch.tensor(val_y,dtype=torch.long,device=device)
 best=None;best_loss=float('inf');stale=0
 for _ in range(epochs):
  model.train()
  for indices in torch.randperm(len(tx),device=device).split(128):
   opt.zero_grad();loss_fn(model(tx[indices]),ty[indices]).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  model.eval()
  with torch.no_grad(): score=loss_fn(model(vx),vy).item()
  if score<best_loss:best_loss=score;best=copy.deepcopy(model.state_dict());stale=0
  else: stale+=1
  if stale>=patience:break
 model.load_state_dict(best);model.eval()
 with torch.no_grad():
  vp=model(vx).softmax(1).cpu().numpy()
  tx_test=torch.tensor(test_x if test_x is not None else val_x,dtype=torch.float32,device=device)
  tp=model(tx_test).softmax(1).cpu().numpy()
 return model,vp,tp,sum(p.numel() for p in model.parameters())
