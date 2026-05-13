#!/usr/bin/env bash
# =============================================================================
# Stage2 正式训练启动脚本: torchrun + 8 卡 DDP + nohup 后台输出到日志
#
# 用法:
#     bash train/train_stage2.sh
#     CONFIG=configs/stage2.yaml bash train/train_stage2.sh
#     EXTRA_OPTS="train.max_steps=20 train.batch_size=1" bash train/train_stage2.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
NGPU=$(awk -F',' '{print NF}' <<< "${GPUS}")

CONFIG="${CONFIG:-configs/stage2.yaml}"
EXTRA_OPTS="${EXTRA_OPTS:-}"
MASTER_PORT="${MASTER_PORT:-29510}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/train_stage2/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"
PID_FILE="${LOG_DIR}/pid.txt"

CMD="${TORCHRUN_BIN} --standalone --nproc_per_node=${NGPU} --master_port=${MASTER_PORT} train/train_stage2.py --config ${CONFIG}"
if [[ -n "${EXTRA_OPTS}" ]]; then
  CMD="${CMD} --opts ${EXTRA_OPTS}"
fi

PYTHON_PATH="$(command -v "${PYTHON_BIN}")"
TORCHRUN_PATH="$(command -v "${TORCHRUN_BIN}")"

{
  echo "============================================================"
  echo " Stage2 training launched"
  echo "   time         : ${TIMESTAMP}"
  echo "   project_root : ${PROJECT_ROOT}"
  echo "   NGPU         : ${NGPU}"
  echo "   GPUS         : ${GPUS}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "   PYTHON_BIN   : ${PYTHON_BIN}"
  echo "   PYTHON_PATH  : ${PYTHON_PATH}"
  echo "   TORCHRUN_BIN : ${TORCHRUN_BIN}"
  echo "   TORCHRUN_PATH: ${TORCHRUN_PATH}"
  echo "   CONFIG       : ${CONFIG}"
  echo "   EXTRA_OPTS   : ${EXTRA_OPTS}"
  echo "   MASTER_PORT  : ${MASTER_PORT}"
  echo "   LOG          : ${LOG_FILE}"
  echo "   CMD          : ${CMD}"
  echo "============================================================"
} | tee "${LOG_DIR}/launch_info.txt"

"${PYTHON_BIN}" - <<'PY'
import importlib, sys
mods = ['torch', 'yaml', 'diffusers']
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, repr(e)))
if missing:
    print('[precheck] missing modules in current python env:')
    for name, err in missing:
        print(f'  - {name}: {err}')
    sys.exit(1)
print('[precheck] python dependency check passed')
PY

nohup bash -c "${CMD}" > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo ""
echo ">>> Training started in background, pid=${TRAIN_PID}"
echo ">>> Tail log:   tail -f ${LOG_FILE}"
echo ">>> Stop it:    kill ${TRAIN_PID}   # or: kill \$(cat ${PID_FILE})"