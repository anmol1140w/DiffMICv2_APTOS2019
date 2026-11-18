#!/usr/bin/env bash
# Real-time monitoring script for the 5-fold cross validation pipeline

LOG_FILE="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/quantum_diffmic_v8/logs/5fold_pipeline.log"
SCREEN_NAME="diffmic_5fold"

clear
echo "============================================================================"
echo "          DiffMIC-v2 5-Fold Cross-Validation Real-Time Monitor              "
echo "============================================================================"
echo " Date: $(date)"
echo " Screen Session: $SCREEN_NAME"
echo " Log File: $LOG_FILE"
echo "============================================================================"
echo ""

echo "[1] GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

echo "[2] Screen Sessions:"
screen -ls || echo "No active screen sessions."
echo ""

echo "[3] Active Docker Containers:"
docker ps --filter "name=diffmic_v8_5fold_runner" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
echo ""

echo "[4] Disk Usage on /dev/nvme0n1p2:"
df -h / | tail -n 1
echo ""

echo "[5] Saved Checkpoints in quantum_diffmic_v8/ckpts:"
ls -lh /home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/quantum_diffmic_v8/ckpts 2>/dev/null || echo "No checkpoints yet."
echo ""

echo "[6] Recent Log Output (Last 25 lines):"
echo "----------------------------------------------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -n 25 "$LOG_FILE"
else
    echo "Log file not created yet."
fi
echo "----------------------------------------------------------------------------"
echo ""
echo "Tip: Run 'tail -f $LOG_FILE' for continuous log streaming."
echo "     Run 'screen -r $SCREEN_NAME' to attach directly to the interactive session."
