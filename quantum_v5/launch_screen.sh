#!/usr/bin/env bash
SCREEN_NAME="quantum_v5"
SCRIPT_PATH="/home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/quantum_v5/run_quantum_pipeline.sh"

if screen -list | grep -q "\.${SCREEN_NAME}"; then
  echo "[-] Screen session '${SCREEN_NAME}' is already active!"
  echo "Use 'screen -r ${SCREEN_NAME}' to attach."
  exit 1
fi

echo "[+] Starting detached screen session '${SCREEN_NAME}'..."
screen -dmS "${SCREEN_NAME}" bash -c "${SCRIPT_PATH}; exec bash"
sleep 2

if screen -list | grep -q "\.${SCREEN_NAME}"; then
  echo "[*] Screen session '${SCREEN_NAME}' successfully launched in background!"
  echo "    Attach command : screen -r ${SCREEN_NAME}"
  echo "    Detach shortcut: Press Ctrl+A followed by D"
  echo "    Live log file  : tail -f /home/pratyum/diffmic_anmol/DiffMICv2_APTOS2019/quantum_v5/logs/unified_training.log"
else
  echo "[!] Failed to launch screen session."
  exit 1
fi
