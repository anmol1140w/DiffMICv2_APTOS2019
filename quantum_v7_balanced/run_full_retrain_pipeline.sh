#!/usr/bin/env bash
set -e

WORKDIR="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019"
cd "$WORKDIR"

mkdir -p quantum_v7_balanced/logs quantum_v7_balanced/ckpts quantum_v7_balanced/outputs

LOG_FILE="quantum_v7_balanced/logs/retrain_pipeline.log"

echo "=========================================================="
echo " STARTING COMPLETE RETRAINING PIPELINE (BS=64, QUANTUM-DEEP)"
echo " Date: $(date)"
echo " Host: $(hostname)"
echo " GPU: NVIDIA RTX PRO 6000 Blackwell (MIG 4g.96gb - 96GB VRAM)"
echo " DCG Backbone: EfficientNet-B3 (Squeeze-and-Excitation Attention)"
echo " Batch Size: 64 (High-Throughput Parallel Processing)"
echo " Quantum: 4-Qubit Entangled Correlators across DCG & Diffusion"
echo " Loss: Class-Balanced Focal Cross-Entropy (Zero Data Duplication)"
echo " Log file: $LOG_FILE"
echo "=========================================================="

GPU_ID="${GPU_DEVICE:-MIG-15c95f22-c7c4-5ca5-99e0-3d5791318c61}"

docker run --rm \
  --name retrain_bs64_container \
  --gpus "\"device=$GPU_ID\"" \
  --shm-size=16g \
  -v /home/pratyum:/home/pratyum \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -u $(id -u):$(id -g) \
  -e HOME=/home/pratyum \
  -e PYTHONUNBUFFERED=1 \
  -w "$WORKDIR" \
  diffmic-env:latest \
  bash -c "
    set -e
    echo ''
    echo '>>> ========================================================================='
    echo '>>> [STAGE 1/3] Retraining Quantum-DCG (BS=64, Max 50 Epochs, Min 20, Class-Balanced Focal Loss)...'
    echo '>>> ========================================================================='
    python3 quantum_v7_balanced/pretrain_dcg_balanced.py \
      --config quantum_v7_balanced/config_balanced.yml \
      --batch-size 64 \
      --epochs 50 \
      --min-epochs 20 \
      --patience 15 \
      --qubits 4 \
      --output quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth

    echo ''
    echo '>>> ========================================================================='
    echo '>>> [STAGE 2/3] Training 512-dim Widened Quantum-DiffMICv2 (BS=64 Effective, Max 150 Epochs)...'
    echo '>>> ========================================================================='
    python3 quantum_v7_balanced/train_balanced.py \
      --config quantum_v7_balanced/config_balanced.yml \
      --aux-checkpoint quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth \
      --max-epochs 150 \
      --min-epochs 40 \
      --batch-size 32 \
      --accumulate-grad-batches 2 \
      --qubits 4 \
      --patience 35

    echo ''
    echo '>>> ========================================================================='
    echo '>>> [STAGE 3/3] Evaluating on Clean 550 Held-Out Test Set (16-Step DDIM + Multimodal ESN)...'
    echo '>>> ========================================================================='
    python3 quantum_v7_balanced/evaluate_clean_test.py \
      --config quantum_v7_balanced/config_balanced.yml \
      --aux-checkpoint quantum_v7_balanced/ckpts/aptos_quantum_dcg_best.pth \
      --output quantum_v7_balanced/outputs/clean_test_results.json \
      --timesteps 16

    echo ''
    echo '=========================================================='
    echo ' ALL STAGES COMPLETED SUCCESSFULLY'
    echo ' Check outputs at: quantum_v7_balanced/outputs/clean_test_results.json'
    echo '=========================================================='
  " 2>&1 | tee "$LOG_FILE"
