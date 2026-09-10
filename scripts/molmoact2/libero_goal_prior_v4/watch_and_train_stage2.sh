#!/usr/bin/env bash
# Wait for v4 Stage1 10k to finish, then auto-launch Stage2 from that checkpoint.
#
# Polls until:
#   1) .../stage1/checkpoints/010000/pretrained_model/config.json exists
#   2) config has num_goal_tokens==8 (refuse accidental v3 Stage1)
#   3) Stage1 train process is no longer holding GPUs (optional but default on)
#
# Then runs train_stage2.sh with POLICY_PATH pointed at Stage1 010000.
#
# Usage:
#   bash scripts/libero_goal_prior_v4/watch_and_train_stage2.sh
#   # or in tmux / nohup:
#   nohup bash scripts/libero_goal_prior_v4/watch_and_train_stage2.sh \
#     >lerobot/outputs/libero_goal_prior_v4/seed_1000/watch_stage2.console.log 2>&1 &
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SEED:-1000}"
POLL_SEC="${POLL_SEC:-60}"
WAIT_STAGE1_IDLE="${WAIT_STAGE1_IDLE:-1}"
REQUIRE_GOAL_TOKENS="${REQUIRE_GOAL_TOKENS:-8}"

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior_v4/seed_${SEED}/stage1}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_OUTPUT_DIR}/checkpoints/010000/pretrained_model}"
STAGE1_CFG="${STAGE1_CKPT}/config.json"
STAGE1_LOG="${STAGE1_LOG:-${WS}/lerobot/outputs/libero_goal_prior_v4/seed_${SEED}/stage1.console.log}"
STAGE2_LOG="${STAGE2_LOG:-${WS}/lerobot/outputs/libero_goal_prior_v4/seed_${SEED}/stage2.console.log}"
WATCH_LOG="${WATCH_LOG:-${WS}/lerobot/outputs/libero_goal_prior_v4/seed_${SEED}/watch_stage2.console.log}"

mkdir -p "$(dirname "${STAGE2_LOG}")"

log() {
  local msg="[v4-watch $(date -Is)] $*"
  echo "${msg}"
  echo "${msg}" >>"${WATCH_LOG}"
}

stage1_train_running() {
  # Match the Stage1 job / script, but not this watcher itself.
  pgrep -af 'molmoact2-goalprior-stage1-v4|libero_goal_prior_v4/train_stage1\.sh|train_stage1\.sh' \
    | grep -v 'watch_and_train_stage2' \
    | grep -v "pgrep" \
    >/dev/null 2>&1
}

stage1_ckpt_ready() {
  [[ -f "${STAGE1_CFG}" ]] || return 1
  python - "${STAGE1_CFG}" "${REQUIRE_GOAL_TOKENS}" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
need = int(sys.argv[2])
got = int(cfg.get("num_goal_tokens") or -1)
src = cfg.get("goal_token_source")
if got != need:
    raise SystemExit(f"num_goal_tokens={got}, need {need}")
if src not in (None, "se3_encoder"):
    # Stage1 should still be se3; warn but allow if already converted somehow.
    pass
print(f"ok goal_tokens={got} source={src} mode={cfg.get('goal_conditioning_mode')}")
PY
}

log "watching Stage1 → ${STAGE1_CKPT}"
log "poll=${POLL_SEC}s wait_idle=${WAIT_STAGE1_IDLE} stage2_log=${STAGE2_LOG}"

while true; do
  if stage1_ckpt_ready; then
    log "Stage1 010000 checkpoint present and goal_tokens=${REQUIRE_GOAL_TOKENS}"
    if [[ "${WAIT_STAGE1_IDLE}" == "1" ]] && stage1_train_running; then
      log "checkpoint ready but Stage1 process still running; waiting for idle..."
    else
      break
    fi
  else
    # Progress breadcrumb from log / last symlink when available.
    last=""
    if [[ -L "${STAGE1_OUTPUT_DIR}/checkpoints/last" ]]; then
      last="$(readlink "${STAGE1_OUTPUT_DIR}/checkpoints/last" || true)"
    fi
    step_hint=""
    if [[ -f "${STAGE1_LOG}" ]]; then
      step_hint="$(tr '\r' '\n' <"${STAGE1_LOG}" | grep -E 'step:[0-9]+K|step:[0-9]+ ' | tail -n 1 | sed -E 's/.*(step:[^ ]+).*/\1/' || true)"
    fi
    log "waiting for 010000 (last=${last:-?} ${step_hint})"
  fi
  sleep "${POLL_SEC}"
done

# Extra settle: let Stage1 finish writing / free GPU memory.
SETTLE_SEC="${SETTLE_SEC:-30}"
log "settling ${SETTLE_SEC}s before Stage2 launch"
sleep "${SETTLE_SEC}"

export SEED
export STAGE1_OUTPUT_DIR
export STAGE1_CKPT
export POLICY_PATH="${POLICY_PATH:-${STAGE1_CKPT}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WS}/lerobot/outputs/libero_goal_prior_v4/seed_${SEED}/stage2}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

log "launching Stage2 POLICY_PATH=${POLICY_PATH}"
log "OUTPUT_DIR=${OUTPUT_DIR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

set -o pipefail
bash "${WS}/scripts/libero_goal_prior_v4/train_stage2.sh" "$@" 2>&1 | tee -a "${STAGE2_LOG}"
rc=${PIPESTATUS[0]}
log "Stage2 exited rc=${rc}"
exit "${rc}"
