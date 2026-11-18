#!/usr/bin/env bash
set -e

WORKDIR="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019"
cd "$WORKDIR"

mkdir -p quantum_v6_resnet/logs quantum_v6_resnet/ckpts quantum_v6_resnet/outputs

LOG_FILE="quantum_v6_resnet/logs/unified_training.log"

echo "=========================================================="
echo " Starting Unified ResNet-18 Quantum-DiffMIC + ESN Pipeline (quantum_v6_resnet)"
echo " Date: $(date)"
echo " Host: $(hostname)"
echo " GPU: NVIDIA RTX PRO 6000 Blackwell"
echo " CNN Backbone: ResNet-18 (512-dim multiscale)"
echo " Quantum: 4-Qubit Entangled Fusion & Cross-Conditioning Bridge"
echo " Prediction Readout: Classical Echo State Network (ESN)"
echo " Log file: $LOG_FILE"
echo "=========================================================="

GPU_ID="${GPU_DEVICE:-MIG-c1433584-b64a-5f40-8301-dbeb8ed10f49}"

docker run --rm \
  --name quantum_v6_resnet_container \
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
    if [ -f 'quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth' ]; then
      echo '>>> [STEP 1/3] Using existing ResNet-18 QuantumDCG checkpoint (quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth)...'
    else
      echo '>>> [STEP 1/3] Pretraining ResNet-18 QuantumDCG (4Q Fusion, 35 Epochs, Batch Size 32)...'
      python3 quantum_v6_resnet/pretrain_dcg_resnet.py \
        --config quantum_v6_resnet/config_resnet.yml \
        --output quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth \
        --epochs 35 \
        --batch-size 32 \
        --qubits 4 \
        --patience 15 \
        --min-epochs 15
    fi

    echo ''
    echo '>>> [STEP 2/3] Training ResNet-18 Quantum-DiffMICv2 with EfficientSAM (Effective Batch 32 [16x2], Max Epochs 1000, Min Epochs 150, Patience 100)...'
    python3 quantum_v6_resnet/train_diffmic_resnet.py \
      --config quantum_v6_resnet/config_resnet.yml \
      --aux-checkpoint quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth \
      --max-epochs 1000 \
      --min-epochs 150 \
      --batch-size 16 \
      --qubits 4 \
      --patience 100 \
      --accumulate-grad-batches 2

    echo ''
    echo '>>> [STEP 3/3] Evaluating Classical ESN Last-Layer Prediction on Trajectories...'
    python3 quantum_v6_resnet/evaluate_esn_last_layer.py \
      --config quantum_v6_resnet/config_resnet.yml \
      --aux-checkpoint quantum_v6_resnet/ckpts/aptos_quantum_dcg_best.pth \
      --output quantum_v6_resnet/outputs/final_esn_results.json \
      --hidden-dim 64 \
      --seed 2026

    echo ''
    echo '=========================================================='
    echo ' ALL STAGES COMPLETED SUCCESSFULLY AT \$(date)'
    echo ' Final Results saved to quantum_v6_resnet/outputs/final_esn_results.json'
    echo '=========================================================='
  " 2>&1 | tee -a "$LOG_FILE"
