#!/usr/bin/env bash
# =============================================================================
# Stage1 正式训练启动脚本: torchrun + 8 卡 DDP + nohup 后台输出到日志
#
# 用法:
#     bash train/train_stage1.sh                       # 直接启动 (使用下方 GPUS 中指定的卡)
#     CONFIG=configs/stage1.yaml bash train/train_stage1.sh
#     EXTRA_OPTS="train.lr=5e-5 data.batch_size=16" bash train/train_stage1.sh
#
# 日志:
#     logs/train_stage1/<timestamp>/train.log  -- stdout + stderr
#     logs/train_stage1/<timestamp>/pid.txt    -- torchrun 的 pid, 便于停止
#
# 停止训练:
#     kill $(cat logs/train_stage1/<timestamp>/pid.txt)
# =============================================================================
set -euo pipefail

# ----------- 切到项目根 (脚本位于 train/ 下, 其父目录即项目根) -----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"


GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
NGPU=$(awk -F',' '{print NF}' <<< "${GPUS}")

# ----------- 其他可配置参数 (仍支持环境变量覆盖) -----------
CONFIG="${CONFIG:-configs/stage1.yaml}"
EXTRA_OPTS="${EXTRA_OPTS:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# ----------- 准备日志目录 -----------
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/train_stage1/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"
PID_FILE="${LOG_DIR}/pid.txt"

# ----------- 组装 torchrun 命令 -----------
CMD="${TORCHRUN_BIN} --standalone --nproc_per_node=${NGPU} --master_port=${MASTER_PORT} train/train_stage1.py --config ${CONFIG}"
if [[ -n "${EXTRA_OPTS}" ]]; then
  CMD="${CMD} --opts ${EXTRA_OPTS}"
fi

PYTHON_PATH="$(command -v "${PYTHON_BIN}")"
TORCHRUN_PATH="$(command -v "${TORCHRUN_BIN}")"

# ----------- 打印 & 落盘启动信息 -----------
{
  echo "============================================================"
  echo " Stage1 training launched"
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
mods = ['torch', 'yaml', 'transformers']
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

# ----------- nohup 后台启动 -----------
nohup bash -c "${CMD}" > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo ""
echo ">>> Training started in background, pid=${TRAIN_PID}"
echo ">>> Tail log:   tail -f ${LOG_FILE}"
echo ">>> Stop it:    kill ${TRAIN_PID}   # or: kill \$(cat ${PID_FILE})"