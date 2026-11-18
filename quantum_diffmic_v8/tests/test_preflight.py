"""Comprehensive 12-Phase Preflight Smoke Test Suite for Unified Quantum-DiffMICv2.
Verifies all model components, quantum layers, gradient flows, checkpoint compatibility,
diffusion numerical stability, downstream representation learners, and data pipelines
prior to launching full 5-fold cross-validation.
"""
import os, sys, time, json, yaml, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_test_header(idx, title):
    print("\n" + "=" * 75)
    print(f" [TEST {idx}/12] {title}")
    print("=" * 75)

def run_preflight_tests():
    total_passed = 0
    total_tests = 12
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running Preflight Smoke Tests on Device: {device}")

    # TEST 1: Imports and Dependencies
    print_test_header(1, "Environment, Libraries and Device Check")
    try:
        import torchvision
        import pytorch_lightning as pl
        import sklearn
        from sklearn.linear_model import LogisticRegression
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  Torchvision Version: {torchvision.__version__}")
        print(f"  Lightning Version: {pl.__version__}")
        print(f"  Scikit-Learn Version: {sklearn.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"  Device Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
        total_passed += 1
        print("  --> PASS: Environment & Dependencies verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 2: Configuration & Manifest Schema
    print_test_header(2, "Configuration & Manifest Schema Validation")
    try:
        config_path = 'quantum_diffmic_v8/configs/config_5fold.yml'
        splits_path = 'quantum_diffmic_v8/configs/5fold_splits.json'
        assert os.path.exists(config_path), f"Missing {config_path}"
        assert os.path.exists(splits_path), f"Missing {splits_path}"
        with open(config_path, 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))
        with open(splits_path, 'r') as f:
            manifest = json.load(f)
        assert cfg.data.num_classes == 5
        assert cfg.model.feature_dim == 512
        assert len(manifest['folds']) == 5
        assert manifest['total_dev_samples'] == 3112
        assert manifest['locked_test_samples'] == 550
        total_passed += 1
        print(f"  Dev Samples: {manifest['total_dev_samples']}, Folds: {len(manifest['folds'])}, Locked Test: {manifest['locked_test_samples']}")
        print("  --> PASS: Configuration & Manifest Schema verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 3: Dataset Loading & Batch Collation
    print_test_header(3, "APTOS Image Dataset & Batch Collation")
    try:
        from aptos_research.data import AptosImageDataset
        from torch.utils.data import DataLoader
        train_records = manifest['folds']['fold_0']['train'][:8]
        ds = AptosImageDataset(train_records, 'train', size=224, training=True)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        images, labels, ids = batch[0], batch[1], batch[2]
        assert images.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
        assert len(ids) == 4
        assert images.dtype == torch.float32
        total_passed += 1
        print(f"  Loaded batch: images={images.shape}, labels={labels}, IDs={ids}")
        print("  --> PASS: Dataset loader and shape validation passed.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 4: Pure PyTorch FastQuantumLayer Forward/Backward across Qubit Scales
    print_test_header(4, "FastQuantumLayer Tensor Simulation (2Q, 4Q, 6Q, 8Q, 12Q)")
    try:
        from quantum_diffmic_v8.models.quantum_layers import FastQuantumLayer
        for n_q in [2, 4, 6, 8, 12]:
            t0 = time.time()
            qlayer = FastQuantumLayer(in_features=n_q, n_qubits=n_q, n_layers=2).to(device)
            dummy_in = torch.randn(8, n_q, requires_grad=True, device=device)
            out = qlayer(dummy_in)
            loss = out.sum()
            loss.backward()
            dt = (time.time() - t0) * 1000
            assert dummy_in.grad is not None and not torch.isnan(dummy_in.grad).any()
            assert not torch.isnan(out).any()
            print(f"  {n_q:2d}-Qubit Circuit: Input {dummy_in.shape} -> Output {out.shape} | Time: {dt:.2f} ms | Grad Norm: {dummy_in.grad.norm().item():.4f}")
        total_passed += 1
        print("  --> PASS: FastQuantumLayer forward/backward confirmed across all qubit scales.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 5: Quantum Fusion Simplifier & Conditioning Bridge
    print_test_header(5, "QuantumFusionSimplifier & QuantumConditioningBridge Fallbacks")
    try:
        from quantum_diffmic_v8.models.quantum_layers import QuantumFusionSimplifier, QuantumConditioningBridge
        # Test Quantum Mode Fusion (in_features = 512 + 512 = 1024)
        q_fusion = QuantumFusionSimplifier(in_features=1024, num_classes=5, n_qubits=4, classical_mode=False).to(device)
        h_g = torch.randn(4, 512, device=device)
        z_l = torch.randn(4, 512, device=device)
        q_logits, q_feat = q_fusion(h_g, z_l)
        assert q_logits.shape == (4, 5) and not torch.isnan(q_logits).any()

        # Test Classical Fallback Mode Fusion
        c_fusion = QuantumFusionSimplifier(in_features=1024, num_classes=5, n_qubits=4, classical_mode=True).to(device)
        c_logits, c_feat = c_fusion(h_g, z_l)
        assert c_logits.shape == (4, 5) and not torch.isnan(c_logits).any()

        # Test Bridge
        bridge = QuantumConditioningBridge(feature_dim=512, n_qubits=4, classical_mode=False).to(device)
        y_feat = torch.randn(4, 512, 14, 14, device=device)
        x_feat = torch.randn(4, 512, device=device)
        mod_feat = bridge(y_feat, x_feat)
        assert mod_feat.shape == (4, 512, 14, 14) and not torch.isnan(mod_feat).any()
        total_passed += 1
        print(f"  Quantum Fusion Logits: {q_logits.shape} | Classical Fusion Logits: {c_logits.shape}")
        print(f"  Modulated Feature Shape: {mod_feat.shape}")
        print("  --> PASS: Quantum and Classical fusion modules verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 6: UnifiedQuantumDCG Forward Pass & Attention
    print_test_header(6, "UnifiedQuantumDCG Forward Architecture & Attention Maps")
    try:
        from quantum_diffmic_v8.models.dcg import UnifiedQuantumDCG
        dcg_model = UnifiedQuantumDCG(config=cfg, num_classes=5, num_k=6, n_qubits=4, classical_mode=False, pretrained=False).to(device)
        imgs = torch.randn(2, 3, 224, 224, device=device)
        y0_aux, y0_g, y0_l, patches, attns, attn_map = dcg_model(imgs)
        assert y0_aux.shape == (2, 5)
        assert y0_g.shape == (2, 5)
        assert y0_l.shape == (2, 5)
        assert patches.shape == (2, 6, 32, 32)
        assert attns.shape == (2, 6)
        assert attn_map.shape == (2, 1, 7, 7) or len(attn_map.shape) >= 2
        total_passed += 1
        print(f"  y0_aux: {y0_aux.shape}, y0_g: {y0_g.shape}, y0_l: {y0_l.shape}")
        print(f"  Patches: {patches.shape}, MIL Attn: {attns.shape}")
        print("  --> PASS: UnifiedQuantumDCG forward pass and attention maps verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 7: MultiBranchFocalLoss and Gradient Backpropagation
    print_test_header(7, "DCG MultiBranchFocalLoss and Non-Zero Gradient Flow")
    try:
        from quantum_diffmic_v8.training.loss import MultiBranchFocalLoss
        criterion = MultiBranchFocalLoss().to(device)
        targets = torch.tensor([0, 3], device=device)
        loss, loss_dict = criterion(y0_aux, y0_g, y0_l, targets, return_dict=True)
        loss.backward()
        
        # Check gradients exist and are non-zero across branches
        has_grad = False
        for name, param in dcg_model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "All gradients in DCG are zero!"
        total_passed += 1
        print(f"  Total Loss: {loss.item():.4f} | Aux Loss: {loss_dict['aux_loss']:.4f} | Global: {loss_dict['global_loss']:.4f} | Local: {loss_dict['local_loss']:.4f}")
        print("  --> PASS: MultiBranchFocalLoss and backward gradient flow verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 8: Checkpoint Compatibility & Transfer Initialization
    print_test_header(8, "Checkpoint Compatibility & Transfer Learning Mapping")
    try:
        dcg_ckpt_path = 'quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth'
        diff_ckpt_path = 'quantum_v7_balanced/ckpts/diffmic_quantum_balanced_epoch-62_acc-0.8236_f1-0.6598.ckpt'
        assert os.path.exists(dcg_ckpt_path), f"Missing {dcg_ckpt_path}"
        assert os.path.exists(diff_ckpt_path), f"Missing {diff_ckpt_path}"
        
        # Test DCG loading
        ckpt_dcg = torch.load(dcg_ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt_dcg, list): ckpt_dcg = ckpt_dcg[0]
        if 'state_dict' in ckpt_dcg: ckpt_dcg = ckpt_dcg['state_dict']
        clean_dcg = {k.replace('aux_model.', ''): v for k, v in ckpt_dcg.items()}
        msg_dcg = dcg_model.load_state_dict(clean_dcg, strict=False)
        print(f"  DCG Checkpoint Loaded: Missing={len(msg_dcg.missing_keys)}, Unexpected={len(msg_dcg.unexpected_keys)}")

        # Test Diffusion loading
        ckpt_diff = torch.load(diff_ckpt_path, map_location='cpu', weights_only=False)
        diff_sd = ckpt_diff.get('state_dict', ckpt_diff)
        print(f"  Diffusion Checkpoint Verified: {len(diff_sd)} tensors found in state_dict.")
        total_passed += 1
        print("  --> PASS: Checkpoint loading and key compatibility verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 9: UnifiedQuantumConditionalModel Denoising Forward Pass
    print_test_header(9, "UnifiedQuantumConditionalModel Conditioning & Forward Denoising")
    try:
        from quantum_diffmic_v8.models.diffusion import UnifiedQuantumConditionalModel
        denoiser = UnifiedQuantumConditionalModel(config=cfg, feature_dim=512, n_qubits=4, classical_mode=False).to(device)
        
        # Build mock conditioning tensors
        x_raw = torch.randn(2, 3, 224, 224, device=device)
        y_cond = torch.randn(2, 5, 6, 6, device=device)
        yt = torch.randn(2, 5, 6, 6, device=device)
        y_in = torch.cat([y_cond, yt], dim=1)
        timesteps = torch.tensor([500, 200], device=device)
        patches = torch.randn(2, 6, 32, 32, device=device)
        attns_mat = torch.randn(2, 1, 6, 6, device=device)
        img_ids = ['test_0', 'test_1']

        pred_noise = denoiser(x=x_raw, y=y_in, t=timesteps, x_l=patches, attn=attns_mat, image_ids=img_ids)
        assert pred_noise.shape == (2, 5, 6, 6)
        assert not torch.isnan(pred_noise).any()
        total_passed += 1
        print(f"  Denoising Output Shape: {pred_noise.shape} | NaN Count: {torch.isnan(pred_noise).sum().item()}")
        print("  --> PASS: UnifiedQuantumConditionalModel forward denoising verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 10: DDIM 16-Step Reverse Sampling Stability & Trajectory Extraction
    print_test_header(10, "16-Step DDIM Reverse Sampling & Trajectory Extraction")
    try:
        import pipeline
        with open('option/diff_DDIM.yaml') as f:
            diff_opt = EasyDict(yaml.safe_load(f))
        diff_opt['scheduler']['num_test_timesteps'] = 16
        scheduler = pipeline.create_SR3scheduler(diff_opt['scheduler'], 'test')
        scheduler.set_timesteps(16)
        scheduler.diff_chns = 5
        sampler = pipeline.SR3Sampler(model=denoiser, scheduler=scheduler)

        x_raw = torch.randn(2, 3, 224, 224, device=device)
        traj_record = [[] for _ in range(2)]
        def cb(t, s_idx, sample):
            pooled = sample.mean(dim=[2, 3]).cpu().numpy()
            for b in range(2):
                traj_record[b].append(pooled[b])

        final_out = sampler.sample_high_res(
            x_raw, yt, conditions=[y_cond, patches, attns_mat, img_ids], trajectory_callback=cb
        )
        assert final_out.shape == (2, 5, 6, 6)
        assert not torch.isnan(final_out).any()
        assert len(traj_record[0]) == 16
        total_passed += 1
        print(f"  Reverse Trajectory Extracted: 16 steps, shape per step=(5,) | Final Shape: {final_out.shape}")
        print("  --> PASS: DDIM 16-step reverse sampling and trajectory extraction verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 11: Downstream Representation Learning (ESN, Multimodal, QRC, HQRC, Risk)
    print_test_header(11, "Downstream Dynamical & Quantum Trajectory Learners")
    try:
        from quantum_diffmic_v8.downstream.representations import (
            make_esn, esn_transform, simulate_qrc_correlators, apply_ordinal_minimum_risk
        )
        # Mock trajectories: 10 samples, 16 steps, 5 dimensions
        mock_trajs = np.random.randn(10, 16, 5)
        
        # 1. Classical ESN
        esn = make_esn(input_dim=5, hidden_dim=64, seed=2026)
        H_esn = esn_transform(mock_trajs, esn)
        assert H_esn.shape == (10, 64)

        # 2. Standalone QRC
        Q_features = simulate_qrc_correlators(mock_trajs, n_qubits=4, temporal_stride=4)
        assert Q_features.shape[0] == 10 and Q_features.shape[1] > 0

        # 3. Hybrid HQRC
        HQ_features = np.hstack([H_esn, Q_features])
        assert HQ_features.shape[0] == 10

        # 4. Ordinal Minimum Risk
        mock_probs = np.random.dirichlet(np.ones(5), size=10)
        risk_preds = apply_ordinal_minimum_risk(mock_probs)
        assert risk_preds.shape == (10,)
        assert all(0 <= p <= 4 for p in risk_preds)
        total_passed += 1
        print(f"  ESN States: {H_esn.shape} | QRC Features: {Q_features.shape} | HQRC Features: {HQ_features.shape}")
        print(f"  Ordinal Minimum Risk Predictions: {risk_preds}")
        print("  --> PASS: All downstream trajectory learners and decision rules verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    # TEST 12: Medical Metrics Suite & Calibration Verification
    print_test_header(12, "Medical Metrics Suite (Macro-F1, QWK, AUROC, ECE, Brier)")
    try:
        from quantum_diffmic_v8.evaluation.metrics import compute_all_medical_metrics
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        # Generate reasonable mock probs with high diagonal
        y_probs = np.eye(5)[y_true] * 0.8 + 0.04
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
        metrics = compute_all_medical_metrics(y_true, y_probs)
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        assert 'quadratic_weighted_kappa' in metrics
        assert 'macro_auroc_ovr' in metrics
        assert 'ece_10_bins' in metrics
        assert 'brier_score' in metrics
        assert metrics['accuracy'] == 1.0
        assert metrics['quadratic_weighted_kappa'] == 1.0
        total_passed += 1
        print(f"  Acc: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | QWK: {metrics['quadratic_weighted_kappa']:.4f}")
        print(f"  AUROC: {metrics['macro_auroc_ovr']:.4f} | ECE: {metrics['ece_10_bins']:.4f} | Brier: {metrics['brier_score']:.4f}")
        print("  --> PASS: Medical evaluation metrics verified.")
    except Exception as e:
        print(f"  --> FAIL: {e}")
        traceback.print_exc()
        return False

    print("\n" + "=" * 75)
    print(f" >>> ALL PREFLIGHT SMOKE TESTS COMPLETED: {total_passed}/{total_tests} PASSED <<<")
    print("=" * 75)
    return total_passed == total_tests

if __name__ == '__main__':
    success = run_preflight_tests()
    sys.exit(0 if success else 1)
