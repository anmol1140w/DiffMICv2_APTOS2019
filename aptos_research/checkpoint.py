"""Checkpoint inventory and shape-safe partial load reporting."""
def inspect_state_dict(model_state, checkpoint_state):
    compatible={}; mismatched={}; unexpected=[]
    for key,value in checkpoint_state.items():
        if key not in model_state: unexpected.append(key)
        elif tuple(value.shape)!=tuple(model_state[key].shape): mismatched[key]={'checkpoint':tuple(value.shape),'model':tuple(model_state[key].shape)}
        else: compatible[key]=value
    missing=sorted(set(model_state)-set(compatible))
    return {'compatible':compatible,'report':{'loaded_keys':sorted(compatible),'missing_keys':missing,'mismatched_shapes':mismatched,'unexpected_keys':sorted(unexpected)}}

def load_compatible(model, checkpoint_state, strict_fraction=None):
    compatible=inspect_state_dict(model.state_dict(),checkpoint_state)
    report=compatible['report']; denom=max(1,len(model.state_dict())); frac=len(compatible['compatible'])/denom
    report['loaded_fraction']=frac
    if strict_fraction is not None and frac<strict_fraction:
        raise ValueError('Only %.1f%% checkpoint keys compatible; required %.1f%%' % (100*frac,100*strict_fraction))
    model.load_state_dict(compatible['compatible'],strict=False)
    return report
