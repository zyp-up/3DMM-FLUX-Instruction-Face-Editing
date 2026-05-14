#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Launch offline DECA parameter extraction across multiple GPUs.
# Usage:
#   NGPU=8 bash scripts/run_extract_multigpu.sh
#
# Environment variables:
#   NGPU        : number of GPUs / shards. Default: 8
#   BATCH_SIZE  : per-GPU batch size. Default: 32
#   NUM_WORKERS : DataLoader workers per GPU. Default: 16
#   EXTRACT_TAG : suffix for logs and PID files. Default: $(date +%Y%m%d_%H%M%S)
# Edit SRC_ROOTS / OUT_ROOTS to choose datasets; entries are matched by index.
# -----------------------------------------------------------------------------
set -euo pipefail

# ====== Project root ======
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ====== Options ======
NGPU="${NGPU:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EXTRACT_TAG="${EXTRACT_TAG:-$(date +%Y%m%d_%H%M%S)}"

# ====== Dataset list (src -> out by index) ======
SRC_ROOTS=(
    "./face_emoji/final_data_raf_bucket_postprocessed"
    "./face_emoji/final_data_v1_bucket_postprocessed"
)
OUT_ROOTS=(
    "${PROJECT_ROOT}/face_emoji/deca_params/raf"
    "${PROJECT_ROOT}/face_emoji/deca_params/v1"
)

if [[ "${#SRC_ROOTS[@]}" -ne "${#OUT_ROOTS[@]}" ]]; then
    echo "[ERROR] SRC_ROOTS and OUT_ROOTS must have the same length" >&2
    exit 1
fi

# ====== Build --src_root / --out_root arguments ======
DATASET_ARGS=()
for i in "${!SRC_ROOTS[@]}"; do
    DATASET_ARGS+=(--src_root "${SRC_ROOTS[$i]}" --out_root "${OUT_ROOTS[$i]}")
done

# ====== Logs and PID files ======
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

# ====== Launch one background shard per GPU ======
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
