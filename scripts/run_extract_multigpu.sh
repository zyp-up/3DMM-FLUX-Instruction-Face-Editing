#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 多卡并行启动 DECA 离线参数提取
# 用法:
#   NGPU=8 bash scripts/run_extract_multi_gpu.sh
#
# 环境变量:
#   NGPU        : 使用的 GPU 卡数 (= num_shards). 默认 8
#   BATCH_SIZE  : 每卡 batch size. 默认 32
#   NUM_WORKERS : 每卡 DataLoader worker 数. 默认 16
#   EXTRACT_TAG : 日志与 PID 文件的后缀, 便于区分多批提取任务. 默认 $(date +%Y%m%d_%H%M%S)
# 修改 SRC_ROOTS / OUT_ROOTS 来指定要提取的数据集 (两个数组一一对应).
# -----------------------------------------------------------------------------
set -euo pipefail

# ====== 项目根目录 ======
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ====== 可调参数 ======
NGPU="${NGPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EXTRACT_TAG="${EXTRACT_TAG:-$(date +%Y%m%d_%H%M%S)}"

# ====== 数据集列表 (src -> out 按下标一一对应) ======
SRC_ROOTS=(
    "./face_emoji/final_data_raf_bucket_postprocessed"
    "./face_emoji/final_data_v1_bucket_postprocessed"
)
OUT_ROOTS=(
    "${PROJECT_ROOT}/face_emoji/deca_params/raf"
    "${PROJECT_ROOT}/face_emoji/deca_params/v1"
)

if [[ "${#SRC_ROOTS[@]}" -ne "${#OUT_ROOTS[@]}" ]]; then
    echo "[ERROR] SRC_ROOTS 与 OUT_ROOTS 长度必须相同" >&2
    exit 1
fi

# ====== 构造 --src_root / --out_root 参数串 ======
DATASET_ARGS=()
for i in "${!SRC_ROOTS[@]}"; do
    DATASET_ARGS+=(--src_root "${SRC_ROOTS[$i]}" --out_root "${OUT_ROOTS[$i]}")
done

# ====== 日志 + PID 目录 ======
LOG_DIR="${PROJECT_ROOT}/logs/extract_${EXTRACT_TAG}"
PID_DIR="${LOG_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

PID_FILE="${LOG_DIR}/all.pids"
: > "${PID_FILE}"

echo "[launch] NGPU=${NGPU}, BATCH_SIZE=${BATCH_SIZE}, NUM_WORKERS=${NUM_WORKERS}"
echo "[launch] EXTRACT_TAG=${EXTRACT_TAG}"
echo "[launch] LOG_DIR=${LOG_DIR}"
echo "[launch] datasets:"
for i in "${!SRC_ROOTS[@]}"; do
    echo "    [$i] ${SRC_ROOTS[$i]}  ->  ${OUT_ROOTS[$i]}"
done

# ====== 逐卡 nohup 后台启动 ======
for ((r=0; r<NGPU; r++)); do
    LOG_FILE="${LOG_DIR}/shard_${r}.log"
    echo "[launch] starting shard ${r}/${NGPU} on GPU ${r}, log=${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${r}" \
    nohup python scripts/extract_deca_params.py \
        "${DATASET_ARGS[@]}" \
        --shard_id "${r}" \
        --num_shards "${NGPU}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "${PID}" > "${PID_DIR}/shard_${r}.pid"
    echo "${r} ${PID}" >> "${PID_FILE}"
done

echo ""
echo "[launch] all ${NGPU} shards launched in background."
echo "[launch] tail a log:       tail -f ${LOG_DIR}/shard_0.log"
echo "[launch] tail all logs:    tail -f ${LOG_DIR}/shard_*.log"
echo "[launch] check processes:  ps -fp \$(awk '{print \$2}' ${PID_FILE})"
echo "[launch] kill all:         kill \$(awk '{print \$2}' ${PID_FILE})"
