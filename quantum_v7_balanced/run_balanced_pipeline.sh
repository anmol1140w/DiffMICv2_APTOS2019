#!/usr/bin/env bash
set -e

WORKDIR="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019"
cd "$WORKDIR"

mkdir -p quantum_v7_balanced/logs quantum_v7_balanced/ckpts quantum_v7_balanced/outputs

LOG_FILE="quantum_v7_balanced/logs/training.log"

echo "=========================================================="
echo " Starting Balanced Quantum-DiffMICv2 Pipeline (quantum_v7)"
echo " Date: $(date)"
echo " Host: $(hostname)"
echo " GPU: NVIDIA RTX PRO 6000 Blackwell"
echo " Diffusion Capacity: 512-dim (Widened)"
echo " Focal Loss Class Weights: [0.8, 2.8, 1.2, 5.0, 2.5] (Zero Data Duplication)"
echo " Global Foundation: EfficientSAM ViT (Cached Fast Path, ~1 min/epoch)"
echo " Log file: $LOG_FILE"
echo "=========================================================="

GPU_ID="${GPU_DEVICE:-MIG-15c95f22-c7c4-5ca5-99e0-3d5791318c61}"

docker run --rm \
  --name quantum_v7_balanced_container \
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
    echo '>>> [STAGE 1/2] Training Balanced Quantum-DiffMICv2 (512-dim, Max 150 Epochs, Patience 35)...'
    python3 quantum_v7_balanced/train_balanced.py \
      --config quantum_v7_balanced/config_balanced.yml \
      --aux-checkpoint quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth \
      --max-epochs 150 \
      --min-epochs 40 \
      --batch-size 16 \
      --qubits 4 \
      --patience 35 \
      --accumulate-grad-batches 2

    echo ''
    echo '>>> [STAGE 2/2] Evaluating on Clean 550 Held-Out Test Set (16-Step DDIM + Multimodal ESN)...'
    python3 quantum_v7_balanced/evaluate_clean_test.py \
      --config quantum_v7_balanced/config_balanced.yml \
      --aux-checkpoint quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth \
      --output quantum_v7_balanced/outputs/clean_test_results.json \
      --timesteps 16

    echo ''
    echo '=========================================================='
    echo ' ALL STAGES COMPLETED SUCCESSFULLY'
    echo '=========================================================='
  " 2>&1 | tee "$LOG_FILE"
