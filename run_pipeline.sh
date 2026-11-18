#!/bin/bash
set -e

echo "=== DiffMIC-v2 + Quantum Research Pipeline ==="
echo "Started at: $(date)"

# Ensure output directories exist
mkdir -p outputs/experiments outputs/training_images outputs/experiments/xai

# Step 1: Trajectory Extraction (Stage 1)
echo ""
echo "============================================================"
echo "--> [1/5] Extracting diffusion trajectories into SQLite..."
echo "============================================================"
python3 -m aptos_research.extract \
  --factory aptos_research.diffmic_adapter:factory \
  --checkpoint /home/pratyum/diffmic/checkpoints/v2/ckpts/aptos-epoch117-acc-0.8186-f1-0.6840.ckpt \
  --cache outputs/trajectories.sqlite \
  --inference-steps 16 \
  --pooling flatten \
  --seed 2026

echo ""
echo "--> Verifying trajectory cache integrity..."
python3 -m aptos_research.cli inspect-cache outputs/trajectories.sqlite

# Step 2: Classical Dynamical Baselines & Controls (Stages 2 & 3)
echo ""
echo "============================================================"
echo "--> [2/5] Running classical dynamical experiments (ESN, GRU, controls)..."
echo "============================================================"
python3 -m aptos_research.experiments \
  --cache outputs/trajectories.sqlite \
  --output outputs/experiments/classical.json \
  --gru \
  --seed 2026

# Step 3: Pre-Registered Evidence Gate (Stage 4)
echo ""
echo "============================================================"
echo "--> [3/5] Evaluating pre-registered bootstrap evidence gate..."
echo "============================================================"
python3 -m aptos_research.cli gate \
  --results outputs/experiments/classical.json \
  --output outputs/experiments/gate.json \
  --minimum-delta 0.01 \
  --alpha 0.05 \
  --bootstrap-samples 5000 \
  --seed 2026 || true

# Step 4: Gated Quantum Reservoir Computing (Stage 5)
echo ""
echo "============================================================"
echo "--> [4/5] Running gated Quantum Reservoir Computing (QRC)..."
echo "============================================================"
python3 -m aptos_research.run_qrc \
  --gate outputs/experiments/gate.json \
  --cache outputs/trajectories.sqlite \
  --output outputs/experiments/qrc.json \
  --qubits 4 \
  --seed 2026 || true

# Step 5: Trustworthiness, Calibration & XAI (Stage 6)
echo ""
echo "============================================================"
echo "--> [5/5] Evaluating calibration, selective risk, and generating XAI plots..."
echo "============================================================"
python3 -m aptos_research.trust_eval \
  --cache outputs/trajectories.sqlite \
  --results outputs/experiments/classical.json \
  --model ordered_esn_k16 \
  --output outputs/experiments/trust.json

python3 -m aptos_research.xai_eval \
  --cache outputs/trajectories.sqlite \
  --sample-id 001639a390f0 \
  --split test \
  --output outputs/experiments/xai

python3 -m aptos_research.xai_eval \
  --cache outputs/trajectories.sqlite \
  --sample-id 8ceff4c4c860 \
  --split train \
  --output outputs/experiments/xai

echo ""
echo "============================================================"
echo "=== Pipeline Completed Successfully at: $(date) ==="
echo "============================================================"
