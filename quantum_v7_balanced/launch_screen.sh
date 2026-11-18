#!/usr/bin/env bash
set -e

SESSION_NAME="retrain_bs64"
SCRIPT_PATH="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/quantum_v7_balanced/run_full_retrain_pipeline.sh"
LOG_FILE="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/quantum_v7_balanced/logs/retrain_pipeline.log"

if screen -list | grep -q "$SESSION_NAME"; then
    echo "[!] Screen session '$SESSION_NAME' already exists! Please kill or attach first:"
    echo "    Attach: screen -r $SESSION_NAME"
    exit 1
fi

echo "[+] Starting detached screen session '$SESSION_NAME'..."
screen -dmS "$SESSION_NAME" bash -c "$SCRIPT_PATH; exec bash"

echo "[*] Screen session '$SESSION_NAME' successfully launched in background!"
echo "    Attach command : screen -r $SESSION_NAME"
echo "    Detach shortcut: Press Ctrl+A followed by D"
echo "    Live log file  : tail -f $LOG_FILE"
