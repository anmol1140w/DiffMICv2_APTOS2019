import argparse, json, time
from pathlib import Path
import numpy as np

from aptos_research.features import FeatureStore
from aptos_research.metrics import classification_metrics
from aptos_research.models import fit_classifier, make_esn, esn_transform
from aptos_research.run_enhanced_qrc import extract_enhanced_qrc

def run_qubit_experiment(X_tr, y_tr, X_va, y_va, X_te, y_te, esn_tr, esn_va, esn_te, n_qubits, temporal_steps=4, seed=2026):
    print(f"\n=======================================================")
    print(f"--> [RUNNING] {n_qubits}-Qubit Quantum Reservoir (Hilbert Dim = {2**n_qubits})")
    print(f"=======================================================")
    t0 = time.time()
    qrc_tr = extract_enhanced_qrc(X_tr, n_qubits=n_qubits, temporal_steps=temporal_steps)
    qrc_va = extract_enhanced_qrc(X_va, n_qubits=n_qubits, temporal_steps=temporal_steps)
    qrc_te = extract_enhanced_qrc(X_te, n_qubits=n_qubits, temporal_steps=temporal_steps)
    extract_time = time.time() - t0
    n_feats = qrc_tr.shape[1]
    print(f"    Quantum simulation done in {extract_time:.1f}s. Quantum features = {n_feats}")

    # 1. Standalone QRC
    clf_qrc = fit_classifier(qrc_tr, y_tr, seed=seed).fit(qrc_tr, y_tr)
    m_qrc_va = classification_metrics(y_va, clf_qrc.predict_proba(qrc_va).tolist())
    m_qrc_te = classification_metrics(y_te, clf_qrc.predict_proba(qrc_te).tolist())

    # 2. Hybrid HQRC
    hyb_tr = np.hstack([esn_tr, qrc_tr])
    hyb_va = np.hstack([esn_va, qrc_va])
    hyb_te = np.hstack([esn_te, qrc_te])

    clf_hyb = fit_classifier(hyb_tr, y_tr, seed=seed).fit(hyb_tr, y_tr)
    m_hyb_va = classification_metrics(y_va, clf_hyb.predict_proba(hyb_va).tolist())
    m_hyb_te = classification_metrics(y_te, clf_hyb.predict_proba(hyb_te).tolist())

    res = {
        'config': {
            'qubits': n_qubits,
            'temporal_steps': temporal_steps,
            'quantum_features_dim': int(n_feats),
            'classical_features_dim': int(esn_tr.shape[1]),
            'hybrid_features_dim': int(hyb_tr.shape[1]),
            'seed': seed,
            'sim_time_sec': round(extract_time, 2)
        },
        'standalone_qrc': {'validation': m_qrc_va, 'test': m_qrc_te},
        'hybrid_hqrc': {'validation': m_hyb_va, 'test': m_hyb_te}
    }
    print(f"    Standalone: Test Acc={m_qrc_te['accuracy']*100:.2f}%, F1={m_qrc_te['macro_f1']:.4f}, QWK={m_qrc_te['quadratic_weighted_kappa']:.4f}, AUROC={m_qrc_te['macro_auroc_ovr']:.4f}")
    print(f"    Hybrid    : Test Acc={m_hyb_te['accuracy']*100:.2f}%, F1={m_hyb_te['macro_f1']:.4f}, QWK={m_hyb_te['quadratic_weighted_kappa']:.4f}, AUROC={m_hyb_te['macro_auroc_ovr']:.4f}")
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='outputs/trajectories.sqlite')
    p.add_argument('--qubits', nargs='+', type=int, default=[10, 12, 14])
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

    print("--> Precomputing Classical ESN (seed 2042)...")
    res_esn = make_esn(X_tr.shape[-1], 64, seed=a.esn_seed)
    esn_tr = esn_transform(X_tr, res_esn)
    esn_va = esn_transform(X_va, res_esn)
    esn_te = esn_transform(X_te, res_esn)

    clf_esn = fit_classifier(esn_tr, y_tr, seed=a.seed).fit(esn_tr, y_tr)
    m_esn_va = classification_metrics(y_va, clf_esn.predict_proba(esn_va).tolist())
    m_esn_te = classification_metrics(y_te, clf_esn.predict_proba(esn_te).tolist())
    print(f"    Classical ESN baseline: Val Acc={m_esn_va['accuracy']*100:.2f}%, Test Acc={m_esn_te['accuracy']*100:.2f}%, F1={m_esn_te['macro_f1']:.4f}, QWK={m_esn_te['quadratic_weighted_kappa']:.4f}")

    sweep_results = {}
    for q in a.qubits:
        res = run_qubit_experiment(X_tr, y_tr, X_va, y_va, X_te, y_te, esn_tr, esn_va, esn_te, n_qubits=q, temporal_steps=a.temporal_steps, seed=a.seed)
        res['classical_esn'] = {'validation': m_esn_va, 'test': m_esn_te}
        out_file = Path(f'outputs/experiments/qrc_{q}qubits.json')
        out_file.write_text(json.dumps(res, indent=2))
        print(f"    Saved to {out_file}")
        sweep_results[q] = res

    # Summary table
    print("\n" + "="*80)
    print(f"{'Qubits':<8} | {'Hilbert Dim':<12} | {'Features':<10} | {'Test Acc':<10} | {'Test F1':<10} | {'Test QWK':<10} | {'AUROC':<10}")
    print("-" * 80)
    for q in a.qubits:
        m = sweep_results[q]['standalone_qrc']['test']
        print(f"{q:<8} | {2**q:<12} | {sweep_results[q]['config']['quantum_features_dim']:<10} | {m['accuracy']*100:.2f}%{'':<4} | {m['macro_f1']:.4f}{'':<4} | {m['quadratic_weighted_kappa']:.4f}{'':<4} | {m['macro_auroc_ovr']:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
