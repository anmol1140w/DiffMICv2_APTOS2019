"""Architecture-valid trajectory plots and perturbation faithfulness helpers."""
def save_probability_trajectory(timesteps, probabilities, output, title='Class probability by reverse timestep'):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for cls in range(probabilities.shape[1]): ax.plot(timesteps,probabilities[:,cls],marker='o',label='grade %d'%cls)
    ax.set(xlabel='Original diffusion timestep',ylabel='Class probability',title=title); ax.legend(ncol=5); fig.tight_layout(); fig.savefig(output,dpi=160); plt.close(fig)

def class_scores_to_probabilities(states, channels, spatial_shape):
    import numpy as np
    x=np.asarray(states,dtype=float)
    h,w=spatial_shape
    if x.ndim!=2 or x.shape[1]!=channels*h*w: raise ValueError('cached state dimension does not match [classes,height,width] metadata')
    scores=x.reshape(len(x),channels,h,w).mean(axis=(2,3)); scores=scores-scores.max(axis=1,keepdims=True)
    exp=np.exp(scores);return exp/exp.sum(axis=1,keepdims=True)

def save_spatial_overlay(image, attention_map, output, alpha=.4):
    """Overlay a real DCG saliency/attention map on the matching RGB input image."""
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    img=np.asarray(image).astype(float)
    if img.ndim!=3 or img.shape[-1]!=3: raise ValueError('image must be RGB HxWx3')
    if img.max()>1: img=img/255.0
    heat=np.asarray(attention_map).squeeze()
    if heat.ndim!=2: raise ValueError('DCG attention map must reduce to HxW')
    heat=np.asarray(Image.fromarray(heat.astype('float32'),mode='F').resize((img.shape[1],img.shape[0]),Image.Resampling.BILINEAR))
    heat=(heat-heat.min())/(heat.max()-heat.min()+1e-12)
    fig,ax=plt.subplots(figsize=(6,6));ax.imshow(np.clip(img,0,1));ax.imshow(heat,cmap='jet',alpha=alpha,vmin=0,vmax=1);ax.axis('off');fig.tight_layout();fig.savefig(output,dpi=160,bbox_inches='tight',pad_inches=0);plt.close(fig)

def leave_one_state_out(states, score_fn, baseline_state=None):
    """Measure target-score change after replacing each observed state in turn."""
    import numpy as np
    sequence=np.asarray(states).copy();base=np.zeros_like(sequence[0]) if baseline_state is None else np.asarray(baseline_state)
    original=float(score_fn(sequence));contributions=[]
    for i in range(len(sequence)):
        altered=sequence.copy();altered[i]=base
        contributions.append({'state_index':i,'score_after_replacement':float(score_fn(altered)),'score_drop':original-float(score_fn(altered))})
    return {'original_score':original,'contributions':contributions}

def perturbation_counterfactual(image, score_fn, saliency, target_score, max_fraction=.5, step=.05, threshold=.5):
    """Greedily neutralize high-saliency regions until a target score is crossed."""
    import numpy as np
    x=np.asarray(image).copy(); sal=np.asarray(saliency)
    spatial=x.ndim==3 and sal.shape==x.shape[-2:]
    if not spatial and sal.shape!=x.shape: raise ValueError('saliency must match image or spatial HxW')
    order=np.argsort(-sal.reshape(-1)); result=x.copy(); total=sal.size
    for frac in np.arange(step,max_fraction+step/2,step):
        n=min(total,int(frac*total)); ix=order[:n]
        if spatial:
            coords=np.unravel_index(ix,sal.shape);result[:,coords[0],coords[1]]=0
        else: result.reshape(-1)[ix]=0
        score=float(score_fn(result))
        if score<=threshold:
            return {'found':True,'fraction_perturbed':float(frac),'score':score,'target_score':float(target_score),'counterfactual':result}
    return {'found':False,'fraction_perturbed':float(max_fraction),'score':float(score_fn(result)),'target_score':float(target_score),'counterfactual':result}

def perturbation_curve(image, score_fn, saliency, fractions=None, insertion=False):
    """Pixel-level deletion/insertion curve using a supplied valid saliency map and score function."""
    import numpy as np
    x=np.asarray(image).copy(); sal=np.asarray(saliency)
    spatial=(x.ndim==3 and sal.shape==x.shape[-2:])
    if spatial:
        order=np.argsort(-sal.reshape(-1)); base=np.zeros_like(x) if insertion else x.copy(); total=sal.size
    elif sal.shape==x.shape:
        order=np.argsort(-sal.reshape(-1));base=np.zeros_like(x).reshape(-1) if insertion else x.reshape(-1).copy();total=sal.size
    else: raise ValueError('saliency must match image dimensions or its spatial HxW map')
    fractions=fractions or [i/20 for i in range(21)]; scores=[]
    for frac in fractions:
        n=int(frac*total); cur=base.copy()
        if spatial:
            h,w=sal.shape; coords=np.unravel_index(order[:n],(h,w))
            if insertion: cur[:,coords[0],coords[1]]=x[:,coords[0],coords[1]]
            else: cur[:,coords[0],coords[1]]=0
        else:
            flat=x.reshape(-1)
            if insertion: cur[order[:n]]=flat[order[:n]]
            else: cur[order[:n]]=0
            cur=cur.reshape(x.shape)
        scores.append(float(score_fn(cur)))
    return fractions,scores
