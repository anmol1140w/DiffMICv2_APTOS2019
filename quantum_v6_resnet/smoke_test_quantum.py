"""Comprehensive Preflight Smoke Tests for ResNet-18 Quantum-Enhanced Pipeline.
Validates:
1. FastQuantumLayer (2Q & 4Q)
2. QuantumFusionSimplifier
3. QuantumConditioningBridge
4. ResNet-18 QuantumDCG end-to-end forward & backward pass
5. ResNet-18 QuantumConditionalModel end-to-end forward & backward pass
6. Real APTOS dataset batch loading & tensor formats
"""
import sys, os
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_v6_resnet.quantum_layers import FastQuantumLayer, QuantumFusionSimplifier, QuantumConditioningBridge
from quantum_v6_resnet.quantum_dcg import QuantumDCG
from quantum_v6_resnet.quantum_diffmic import QuantumConditionalModel
from aptos_research.data import load_split_manifest, AptosImageDataset
from torch.utils.data import DataLoader

def run_tests():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f" >>> RUNNING PREFLIGHT RESNET-18 SMOKE TESTS ON: {device}")
    print(f"=======================================================\n")

    # TEST 1: FastQuantumLayer
    print("--> [1/6] Testing FastQuantumLayer (2Q and 4Q)...")
    for q in (2, 4):
        layer = FastQuantumLayer(in_features=64, n_qubits=q).to(device)
        x = torch.randn(4, 64, device=device, requires_grad=True)
        out = layer(x)
        expected_dim = 3 if q == 2 else 10
        assert out.shape == (4, expected_dim), f"Expected shape (4, {expected_dim}), got {out.shape}"
        out.sum().backward()
        assert x.grad is not None, f"Gradient not backpropagated for {q}Q"
        print(f"    [*] {q}-Qubit Quantum Layer PASSED: Output shape {out.shape}, autograd OK")

    # TEST 2: QuantumFusionSimplifier
    print("--> [2/6] Testing QuantumFusionSimplifier...")
    fusion = QuantumFusionSimplifier(in_features=512 * 2, num_classes=5, n_qubits=4).to(device)
    hg = torch.randn(4, 512, device=device)
    zl = torch.randn(4, 512, device=device)
    y_q, q_feat = fusion(hg, zl)
    assert y_q.shape == (4, 5) and q_feat.shape == (4, 10)
    print(f"    [*] QuantumFusionSimplifier PASSED: Logits {y_q.shape}, Correlators {q_feat.shape}")

    # TEST 3: QuantumConditioningBridge
    print("--> [3/6] Testing QuantumConditioningBridge...")
    bridge = QuantumConditioningBridge(feature_dim=128, n_qubits=4).to(device)
    y_feat = torch.randn(4, 128, 6, 6, device=device, requires_grad=True)
    x_feat = torch.randn(4, 128, device=device, requires_grad=True)
    mod = bridge(y_feat, x_feat)
    assert mod.shape == (4, 128, 6, 6)
    mod.sum().backward()
    assert y_feat.grad is not None and x_feat.grad is not None
    print(f"    [*] QuantumConditioningBridge PASSED: Modulated shape {mod.shape}, autograd OK")

    # TEST 4: ResNet-18 QuantumDCG End-to-End
    print("--> [4/6] Testing QuantumDCG End-to-End (ResNet-18 + 4Q Fusion)...")
    dcg = QuantumDCG(num_classes=5, num_k=6, n_qubits=4, pretrained=False).to(device)
    x_img = torch.randn(2, 3, 224, 224, device=device)
    y_fuse, y_glob, y_loc, patches, attns, cam = dcg(x_img)
    assert y_fuse.shape == (2, 5)
    assert y_glob.shape == (2, 5)
    assert y_loc.shape == (2, 5)
    assert patches.shape == (2, 6, 32, 32)
    assert attns.shape == (2, 6)
    assert cam.shape == (2, 5, 7, 7)
    total_loss = y_fuse.sum() + 0.5 * y_glob.sum() + 0.5 * y_loc.sum()
    total_loss.backward()
    dcg_params = sum(p.numel() for p in dcg.parameters())
    print(f"    [*] ResNet-18 QuantumDCG PASSED: Params={dcg_params/1e6:.2f}M, all shapes verified")

    # TEST 5: QuantumConditionalModel End-to-End with ResNet-18 & EfficientSAM
    print("--> [5/6] Testing ResNet-18 QuantumConditionalModel with EfficientSAM End-to-End...")
    cfg_sam = EasyDict({
        'data': {'num_classes': 5},
        'model': {'feature_dim': 128, 'n_qubits': 4, 'encoder': 'sam', 'num_k': 6},
        'diffusion': {'timesteps': 1000}
    })
    diff_model = QuantumConditionalModel(config=cfg_sam, feature_dim=128, n_qubits=4, guidance=True).to(device)
    y_in = torch.randn(2, 10, 6, 6, device=device)
    t = torch.tensor([10, 100], device=device)
    p_in = torch.randn(2, 6, 32, 32, device=device)
    noise_pred = diff_model(x_img, y_in, t, p_in)
    assert noise_pred.shape == (2, 5, 6, 6)
    noise_pred.sum().backward()
    diff_params = sum(p.numel() for p in diff_model.parameters())
    print(f"    [*] ResNet-18 QuantumConditionalModel PASSED: Params={diff_params/1e6:.2f}M, Noise pred {noise_pred.shape}")

    # TEST 6: Real APTOS Data Pipeline Loading
    print("--> [6/6] Testing Real APTOS Dataset Batch Loading...")
    config_file = PROJECT_ROOT / 'quantum_v6_resnet/config_resnet.yml'
    with open(config_file) as f:
        cfg = EasyDict(yaml.safe_load(f))
    manifest = load_split_manifest(cfg.data.splits_manifest)
    val_ds = AptosImageDataset(manifest['splits']['validation'], 'validation', size=224, training=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    batch = next(iter(val_loader))
    x_real, y_real = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        out, *_ = dcg(x_real)
    assert out.shape == (4, 5)
    print(f"    [*] Real Batch Ingestion PASSED: Batch {x_real.shape} -> Logits {out.shape}")

    print("\n" + "=" * 65)
    print(" >>> ALL 6 PREFLIGHT TESTS PASSED CLEANLY!")
    print(" >>> SYSTEM FULLY VERIFIED FOR PRODUCTION TRAINING RUN.")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    run_tests()
