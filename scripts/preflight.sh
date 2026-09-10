#!/usr/bin/env bash
# Check the environment before spending GPU hours. Run this first.
#   bash scripts/preflight.sh                  # evaluation prerequisites
#   bash scripts/preflight.sh /path/to/ckpt    # also check that checkpoint
#   TRAIN=1 bash scripts/preflight.sh          # also check training prerequisites
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

fail=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
note() { printf '        %s\n' "$1"; }

echo "Paths"
if lit_need LIT_MOLMOACT2 "the Molmoact2 checkout" "scripts/libero_eval" 2>/dev/null; then
  ok "LIT_MOLMOACT2 = $LIT_MOLMOACT2"
  if [ -f "$LIT_MOLMOACT2/lerobot/src/lerobot/policies/molmoact2/modeling_molmoact2.py" ]; then
    ok "lerobot submodule is checked out"
    sub="$(git -C "$LIT_MOLMOACT2/lerobot" rev-parse --short HEAD 2>/dev/null || echo '?')"
    [ "$sub" = "b966b8c0" ] || [ "${sub:0:7}" = "b966b8c" ] \
      && ok "lerobot at $sub (paper commit)" \
      || note "lerobot at $sub — the paper used b966b8c0; run: git submodule update --init"
  else
    bad "lerobot submodule is empty — clone with --recursive, or: git submodule update --init"
  fi
else
  bad "LIT_MOLMOACT2 unset or wrong"
fi

if lit_need LIBERO_PLUS_ROOT "the LIBERO-plus checkout" \
     "libero/libero/benchmark/task_classification.json" 2>/dev/null; then
  ok "LIBERO_PLUS_ROOT = $LIBERO_PLUS_ROOT"
  n=$(python3 - "$LIBERO_PLUS_ROOT" <<'PY' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]+"/libero/libero/benchmark/task_classification.json"))
print(sum(len(v) for v in d.values()))
PY
)
  [ "$n" = "10030" ] && ok "task_classification.json lists 10030 tasks" \
                     || bad "task_classification.json lists ${n:-?} tasks, expected 10030"
  [ -e "$LIBERO_PLUS_ROOT/libero/libero/assets" ] && ok "LIBERO-plus assets present" \
    || bad "LIBERO-plus assets missing (libero/libero/assets)"
else
  bad "LIBERO_PLUS_ROOT unset or wrong"
fi

if [ "${TRAIN:-0}" = "1" ]; then
  if lit_need DATASET_ROOT "LIBERO in LeRobot format" 2>/dev/null; then
    ok "DATASET_ROOT = $DATASET_ROOT"
  else
    bad "DATASET_ROOT unset or wrong (training only)"
  fi
fi

echo
echo "Runtime"
if command -v nvidia-smi >/dev/null; then
  g=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  ok "$g GPU(s) visible"
  [ "$g" -ge 4 ] || note "the 8-GPU launchers assume 8; set EVAL_GPU_IDS to match your machine"
else
  bad "nvidia-smi not found"
fi
[ -f "${LIT_MOLMOACT2:-/nonexistent}/scripts/activate_train_env.sh" ] \
  && ok "activate_train_env.sh present" || bad "activate_train_env.sh missing"

CK="${1:-}"
if [ -n "$CK" ]; then
  echo
  echo "Checkpoint"
  if [ -f "$CK/config.json" ]; then
    ok "config.json"
    ls "$CK"/*.safetensors >/dev/null 2>&1 && ok "weights present" || bad "no .safetensors in $CK"
    python3 - "$CK" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]+"/config.json"))
keys=("enable_goal_pose","mask_image_from_action_expert","semantic_visual_recurrent",
      "enable_ae_pose_head","pose_recon_loss_weight","target_pose_delta_index")
have={k:c[k] for k in keys if k in c}
print("        LIT switches:", have if have else "(none — this looks like a baseline checkpoint)")
PY
  else
    bad "no config.json in $CK — point at the pretrained_model directory"
  fi
fi

echo
[ "$fail" = 0 ] && echo "preflight passed." || { echo "preflight FAILED — fix the above first."; exit 1; }
