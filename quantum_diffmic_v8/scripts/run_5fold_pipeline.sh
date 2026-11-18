#!/usr/bin/env bash
set -e

WORKDIR="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019"
cd "$WORKDIR"

mkdir -p quantum_diffmic_v8/logs quantum_diffmic_v8/ckpts quantum_diffmic_v8/outputs

LOG_FILE="quantum_diffmic_v8/logs/5fold_pipeline.log"

echo "============================================================================" | tee -a "$LOG_FILE"
echo " Starting Full Stratified 5-Fold Cross-Validation Pipeline" | tee -a "$LOG_FILE"
echo " Architecture: Unified Quantum-DiffMICv2 (quantum_diffmic_v8)" | tee -a "$LOG_FILE"
echo " Host: $(hostname) | Date: $(date)" | tee -a "$LOG_FILE"
echo " Effective Batch Size: 64 (Physical 32 x Accumulation 2)" | tee -a "$LOG_FILE"
echo " Quantum Circuit: FastQuantumLayer (4 Qubits, Vectorized Ry + Circular CNOT)" | tee -a "$LOG_FILE"
echo " Guidance: UnifiedQuantumDCG (EfficientNet-B3 + Fast GPU Regional Proposal)" | tee -a "$LOG_FILE"
echo " Denoiser: 512-dim Multi-Scale Conditioning + Quantum FiLM Bridge" | tee -a "$LOG_FILE"
echo " Downstream: ESN, Multimodal ESN, Standalone QRC, Hybrid HQRC, Bayes Risk" | tee -a "$LOG_FILE"
echo " Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"

GPU_ID="${GPU_DEVICE:-MIG-15c95f22-c7c4-5ca5-99e0-3d5791318c61}"
CONTAINER_NAME="diffmic_v8_5fold_runner"

# Ensure no orphan container with the same name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "[+] Launching Docker Container $CONTAINER_NAME on GPU $GPU_ID..." | tee -a "$LOG_FILE"

docker run --rm \
  --name "$CONTAINER_NAME" \
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
    echo '>>> Container Environment Initialized.'
    echo '>>> Python Path:' \$(which python3)
    echo '>>> Current Directory:' \$(pwd)
    
    python3 quantum_diffmic_v8/cross_validation/orchestrator.py \
      --config quantum_diffmic_v8/configs/config_5fold.yml \
      --splits quantum_diffmic_v8/configs/5fold_splits.json \
      --dcg-epochs 20 \
      --diff-epochs 35 \
      --batch-size 32 \
      --accumulate-grad-batches 2 \
      --qubits 4 \
      --start-fold 0 \
      --num-folds 5
  " 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo " 5-FOLD CROSS-VALIDATION PIPELINE COMPLETED SUCCESSFULLY: $(date)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
