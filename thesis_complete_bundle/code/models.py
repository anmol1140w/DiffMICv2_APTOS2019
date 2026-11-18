"""Static, permutation-invariant, classical recurrent and gated QRC models."""
def pool_trajectory(states, method='mean'):
    import numpy as np
    x=np.asarray(states,dtype=float)
    if x.ndim!=3: raise ValueError('expected [N,K,D]')
    if method=='mean': return x.mean(axis=1)
    if method=='mean_std': return np.concatenate([x.mean(axis=1),x.std(axis=1)],axis=1)
    if method=='flatten_unordered': return np.sort(x,axis=1).reshape(x.shape[0],-1)
    raise ValueError(method)

def make_esn(input_dim, hidden_dim=64, seed=0, spectral_radius=.9, connectivity=.1, leak=.5):
    import numpy as np
    rng=np.random.default_rng(seed); w=rng.normal(0,1,(hidden_dim,hidden_dim)); w[rng.random(w.shape)>connectivity]=0
    radius=max(abs(np.linalg.eigvals(w)))
    return {'input':rng.normal(0,.5,(input_dim,hidden_dim)), 'reservoir':w*(spectral_radius/(radius or 1)), 'leak':float(leak), 'spectral_radius':spectral_radius,'connectivity':connectivity}

def esn_transform(states, reservoir):
    import numpy as np
    x=np.asarray(states,dtype=float); outs=[]
    for sample in x:
        h=np.zeros(reservoir['reservoir'].shape[0])
        for u in sample: h=(1-reservoir['leak'])*h+reservoir['leak']*np.tanh(u@reservoir['input']+h@reservoir['reservoir'])
        outs.append(np.concatenate([h, sample[-1]]))
    return np.asarray(outs)

def qrc_features(states, n_qubits=4, seed=0, shots=0, noise=0.0):
    """Small state-vector reservoir: angle encode each compact state and apply fixed entangling layers.

    Returned Z expectation features include encoding/simulator settings in metadata. This is a research
    implementation, not evidence of quantum advantage; noisy finite-shot readout is optional.
    """
    import numpy as np
    if n_qubits<1 or n_qubits>10: raise ValueError('state-vector QRC supports 1..10 qubits')
    if not 0.0 <= noise <= 1.0: raise ValueError('depolarizing probability must be in [0,1]')
    rng=np.random.default_rng(seed); dim=2**n_qubits; out=[]
    for sequence in np.asarray(states,dtype=float):
        state=np.zeros(dim,dtype=complex); state[0]=1
        for vector in sequence:
            # Deterministically pool compact classical coordinates to each qubit angle.
            angles=np.asarray([vector[q::n_qubits].mean() for q in range(n_qubits)])
            angles=np.pi*np.tanh(angles)
            # Product RY encoding
            for q,theta in enumerate(angles):
                bit=1<<q; new=state.copy(); c=np.cos(theta/2); s=np.sin(theta/2)
                for i in range(dim):
                    if i&bit: continue
                    a,b=state[i],state[i|bit]; new[i]=c*a-s*b; new[i|bit]=s*a+c*b
                state=new
            # Fixed ring entanglement (CNOTs) is unitary and introduces cross-qubit mixing.
            for q in range(n_qubits):
                control=1<<q; target=1<<((q+1)%n_qubits)
                for i in range(dim):
                    if i&control and not i&target:
                        j=i|target; state[i],state[j]=state[j],state[i]
        probs=np.abs(state)**2; z=[]
        for q in range(n_qubits):
            z.append(sum((1 if not i&(1<<q) else -1)*probs[i] for i in range(dim)))
        # Simple readout depolarization model shrinks Pauli-Z expectations.
        z=np.asarray(z)*(1.0-noise)
        if shots:
            z=np.asarray([2*rng.binomial(int(shots),(1+float(v))/2)/int(shots)-1 for v in z])
        out.append(z)
    return np.asarray(out), {'n_qubits':n_qubits,'encoding':'per-state RY angle encoding; mean-pool coordinates per qubit; angle=pi*tanh(pooled value)','reservoir':'fixed ring CNOT layer after each state','readout':'single-qubit Pauli-Z expectations','shots':int(shots),'readout_depolarizing_probability':float(noise),'simulator':'numpy state-vector'}

def fit_classifier(features, labels, seed=0, max_iter=1000):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=max_iter,class_weight=None,random_state=seed))

def make_gru(input_dim, hidden_dim=64, classes=5):
    import torch
    from torch import nn
    return nn.GRU(input_dim,hidden_dim,batch_first=True), nn.Linear(hidden_dim,classes)
