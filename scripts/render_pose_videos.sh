#!/usr/bin/env bash
# LIT 位姿解码叠加视频：ID（LIBERO 干净任务）+ OOD（LIBERO-Plus 变体）。GPU 5/6/7 各跑一份清单。
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"; lit_need_eval || exit 2
M=$LIT_MOLMOACT2; CK="${1:?usage: render_pose_videos.sh <checkpoint>}"
OUT=$LIT_WORK/pose_videos; mkdir -p "$OUT"
cd "$M" && source scripts/activate_train_env.sh >/dev/null 2>&1
export LIBERO_RESOURCE_ROOT="${LIBERO_RESOURCE_ROOT:-$M}" LIBERO_PLUS_ROOT LIBERO_PLUS_FIX_LANG=1
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# 每行: gpu envtype suite task_id label
JOBS="
5 libero      libero_spatial 0    id_spatial
5 libero_plus libero_spatial 608  ood_spatial_camera
5 libero_plus libero_spatial 2110 ood_spatial_light
5 libero_plus libero_spatial 0    ood_spatial_background
6 libero      libero_object  0    id_object
6 libero_plus libero_object  646  ood_object_camera
6 libero_plus libero_object  2221 ood_object_light
6 libero_plus libero_object  1818 ood_object_distractor
7 libero      libero_goal    0    id_goal
7 libero      libero_10      0    id_long
7 libero_plus libero_spatial 1725 ood_spatial_distractor
7 libero_plus libero_object  0    ood_object_background
"
run_gpu() {
  local gpu=$1
  echo "$JOBS" | awk -v g=$gpu '$1==g' | while read -r g et suite tid label; do
    d=$OUT/$label; mkdir -p "$d"
    [ -f "$d/eval_info.json" ] && { echo "[skip] $label"; continue; }
    if [ "$et" = libero_plus ]; then PP="$LIBERO_PLUS_ROOT:$M/lerobot/src"; CFG=$LIT_WORK/libero_config_plus; else PP="$M/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"; CFG=""; fi
    env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g POSE_VIZ_DIR=$d PYTHONPATH="$PP" ${CFG:+LIBERO_CONFIG_PATH=$CFG} \
    python -m lerobot.scripts.lerobot_eval \
      --policy.path="$CK" --policy.device=cuda --policy.inference_action_mode=continuous \
      --policy.disable_visual_input=false --policy.per_episode_seed=true --policy.eval_seed=1000 \
      --policy.enable_inference_cuda_graph=false \
      --env.type=$et --env.task="$suite" --env.task_ids="[$tid]" --env.control_mode=relative --env.max_parallel_tasks=1 \
      --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false --eval.max_episodes_rendered=1 \
      --seed=1000 --output_dir="$d" > "$d/eval.log" 2>&1
    ok=$(python3 -c "import json;print(json.load(open('$d/eval_info.json'))['per_task'][0]['metrics']['successes'][0])" 2>/dev/null)
    echo "[$(date '+%T')] gpu$g $label rc=$? success=${ok:-?} mp4=$(find $d -name '*.mp4' | wc -l)"
  done
}
run_gpu 5 & run_gpu 6 & run_gpu 7 & wait
echo "ALL DONE"; find "$OUT" -name '*.mp4' | sort
