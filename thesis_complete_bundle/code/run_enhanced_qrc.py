"""Enhanced Quantum Reservoir Computing (QRC) & Hybrid Quantum-Classical Reservoir (HQRC).
Evaluates 1-body expectations <Z_i> and 2-body entanglement correlators <Z_i Z_j> across
diffusion trajectory timesteps, and computes Hybrid (ESN + QRC) ensembles.
"""
import argparse, json, time
from pathlib import Path
import numpy as np

from aptos_research.features import FeatureStore
from aptos_research.metrics import classification_metrics
from aptos_research.models import fit_classifier, make_esn, esn_transform
from aptos_research.registry import record

def extract_enhanced_qrc(states, n_qubits=6, temporal_steps=4):
    """Simulates an n-qubit quantum reservoir with ring CNOT entanglement,
    measuring single-qubit <Z_i> and two-qubit correlators <Z_i Z_j> at selected timesteps.
    """
    dim = 2**n_qubits
    all_features = []
    T = states.shape[1]
    measure_indices = np.linspace(T//temporal_steps - 1, T - 1, temporal_steps, dtype=int)
    
    # Precompute observable projection matrices
    indices = np.arange(dim)
    mask_1b = np.where((indices[:, None] & (1 << np.arange(n_qubits))) == 0, 1.0, -1.0)
    pairs = [(q1, q2) for q1 in range(n_qubits) for q2 in range(q1 + 1, n_qubits)]
    mask_2b = np.empty((dim, len(pairs)), dtype=np.float64)
    for idx, (q1, q2) in enumerate(pairs):
        b1 = (indices & (1 << q1)) > 0
        b2 = (indices & (1 << q2)) > 0
        mask_2b[:, idx] = np.where(b1 == b2, 1.0, -1.0)
    mask_all = np.hstack([mask_1b, mask_2b]) # (dim, n_qubits + pairs)

    for sequence in states:
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        sample_obs = []
        
        for t_step, vector in enumerate(sequence):
            # Deterministic angle encoding
            angles = np.asarray([vector[q::n_qubits].mean() for q in range(n_qubits)])
            angles = np.pi * np.tanh(angles)
            
            # Product RY rotations
            for q, theta in enumerate(angles):
                bit = 1 << q; new = state.copy(); c = np.cos(theta / 2.0); s = np.sin(theta / 2.0)
                for i in range(dim):
                    if i & bit: continue
                    a, b = state[i], state[i | bit]
                    new[i] = c * a - s * b
                    new[i | bit] = s * a + c * b
                state = new
                
            # Cyclic Ring CNOT Entanglement
            for q in range(n_qubits):
                control = 1 << q; target = 1 << ((q + 1) % n_qubits)
                for i in range(dim):
                    if i & control and not i & target:
                        j = i | target
                        state[i], state[j] = state[j], state[i]
                        
            # Measure observables at selected diffusion checkpoints
            if t_step in measure_indices:
                probs = np.abs(state)**2
                obs = probs @ mask_all
                sample_obs.extend(obs.tolist())
                        
        all_features.append(sample_obs)
    return np.asarray(all_features, dtype=np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='outputs/trajectories.sqlite')
    p.add_argument('--output', default='outputs/experiments/qrc_enhanced.json')
    p.add_argument('--qubits', type=int, default=6)
    p.add_argument('--temporal-steps', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--esn-seed', type=int, default=2042)
    a = p.parse_args()

    store = FeatureStore(a.cache)
    tr, va, te = store.load_split('train'), store.load_split('validation'), store.load_split('test')
    
    def seq(rows): return np.stack([x['states'] for x in rows])
    X_tr, y_tr = seq(tr), [x['label'] for x in tr]
    X_va, y_va = seq(va), [x['label'] for x in va]
    X_te, y_te = seq(te), [x['label'] for x in te]

    print(f"--> [1/3] Extracting {a.qubits}-qubit Quantum Reservoir features with 2-body correlators across {a.temporal_steps} timesteps...")
    t0 = time.time()
    qrc_tr = extract_enhanced_qrc(X_tr, a.qubits, a.temporal_steps)
    qrc_va = extract_enhanced_qrc(X_va, a.qubits, a.temporal_steps)
    qrc_te = extract_enhanced_qrc(X_te, a.qubits, a.temporal_steps)
    q_time = time.time() - t0
    print(f"    Done in {q_time:.1f}s. Quantum feature dimension = {qrc_tr.shape[1]}")

    # 1. Standalone Quantum Reservoir
    print("--> [2/3] Fitting Standalone Quantum Reservoir classifier...")
    clf_qrc = fit_classifier(qrc_tr, y_tr, seed=a.seed).fit(qrc_tr, y_tr)
    m_qrc_va = classification_metrics(y_va, clf_qrc.predict_proba(qrc_va).tolist())
    m_qrc_te = classification_metrics(y_te, clf_qrc.predict_proba(qrc_te).tolist())

    # 2. Classical ESN
    print("--> [3/3] Fitting Classical ESN & Hybrid Quantum-Classical Reservoir (HQRC)...")
    reservoir = make_esn(X_tr.shape[-1], 64, seed=a.esn_seed)
    esn_tr = esn_transform(X_tr, reservoir)
    esn_va = esn_transform(X_va, reservoir)
    esn_te = esn_transform(X_te, reservoir)

    clf_esn = fit_classifier(esn_tr, y_tr, seed=a.seed).fit(esn_tr, y_tr)
    m_esn_va = classification_metrics(y_va, clf_esn.predict_proba(esn_va).tolist())
    m_esn_te = classification_metrics(y_te, clf_esn.predict_proba(esn_te).tolist())

    # 3. Hybrid Quantum-Classical Ensemble
    hyb_tr = np.hstack([esn_tr, qrc_tr])
    hyb_va = np.hstack([esn_va, qrc_va])
    hyb_te = np.hstack([esn_te, qrc_te])

    clf_hyb = fit_classifier(hyb_tr, y_tr, seed=a.seed).fit(hyb_tr, y_tr)
    m_hyb_va = classification_metrics(y_va, clf_hyb.predict_proba(hyb_va).tolist())
    m_hyb_te = classification_metrics(y_te, clf_hyb.predict_proba(hyb_te).tolist())

    results = {
        'config': {
            'qubits': a.qubits,
            'temporal_steps': a.temporal_steps,
            'quantum_features_dim': int(qrc_tr.shape[1]),
            'classical_features_dim': int(esn_tr.shape[1]),
            'hybrid_features_dim': int(hyb_tr.shape[1]),
            'seed': a.seed
        },
        'standalone_qrc': {'validation': m_qrc_va, 'test': m_qrc_te},
        'classical_esn': {'validation': m_esn_va, 'test': m_esn_te},
        'hybrid_hqrc': {'validation': m_hyb_va, 'test': m_hyb_te}
    }

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + '\n')
    record('outputs/experiments/registry.jsonl', 'enhanced-qrc', 'COMPLETED', results['config'], {'output': str(out_path.resolve())}, {'test_qrc_acc': m_qrc_te['accuracy'], 'test_hyb_acc': m_hyb_te['accuracy']})
    print(f"\nSaved results to {a.output}")

    print("\n==================== COMPARISON SUMMARY ====================")
    print(f"{'Architecture':<32} | {'Test Acc':<8} | {'Test F1':<8} | {'Test Kappa':<10}")
    print("-" * 65)
    print(f"{'1. Standalone Quantum (6-Qubit QRC)':<32} | {m_qrc_te['accuracy']*100:6.2f}% | {m_qrc_te['macro_f1']:7.4f} | {m_qrc_te['quadratic_weighted_kappa']:8.4f}")
    print(f"{'2. Classical ESN Reservoir':<32} | {m_esn_te['accuracy']*100:6.2f}% | {m_esn_te['macro_f1']:7.4f} | {m_esn_te['quadratic_weighted_kappa']:8.4f}")
    print(f"{'3. Hybrid Quantum-Classical (HQRC)':<32} | {m_hyb_te['accuracy']*100:6.2f}% | {m_hyb_te['macro_f1']:7.4f} | {m_hyb_te['quadratic_weighted_kappa']:8.4f}")
    print("============================================================\n")

if __name__ == '__main__':
    main()
