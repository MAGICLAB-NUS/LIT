#!/usr/bin/env bash
# A800 回收前，把「可复现论文所需的最终 checkpoint」从 A800j 本地盘搬到 /pfs（fm1 可读）。
# 只搬每个 run 的末步，不搬中间步与 wandb/videos。
set -uo pipefail
DST=/pfs/pfs-Tr4Uts/JM_work/paper_ckpt
LOG=/data2/JM/migrate_ckpt.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
cp1() {  # $1=源目录 $2=目标相对路径
  local src="$1" rel="$2"
  [ -e "$src" ] || { say "SKIP 缺失 $rel"; return; }
  mkdir -p "$DST/$rel"
  rsync -a --info=stats2 "$src/" "$DST/$rel/" >/dev/null 2>&1 \
    && say "OK   $rel  ($(du -sh "$DST/$rel" | cut -f1))" || say "FAIL $rel"
}
say "=== 迁移开始 ==="
O=/data2/JM/Code/molmoact2/lerobot/outputs
# --- MolmoAct2：主表两行 + Stage 1 + 五个消融臂 ---
cp1 $O/libero_goal_prior_v3/seed_1000/baseline_bs224/checkpoints/030000/pretrained_model molmoact2/baseline_030000
cp1 $O/libero_goal_prior_v3/seed_1000/stage1/checkpoints/010000/pretrained_model        molmoact2/lit_stage1_010000
cp1 $O/libero_goal_prior_v3/seed_1000/stage2/checkpoints/030000/pretrained_model        molmoact2/lit_full_stage2_030000
cp1 $O/libero_ablation/sw_stagewise/stage1/checkpoints/010000/pretrained_model          molmoact2/abl1_stagewise_stage1_010000
cp1 $O/libero_ablation/sw_stagewise/stage2/checkpoints/030000/pretrained_model          molmoact2/abl1_stagewise_stage2_030000
cp1 $O/libero_ablation/abl_wo_stage1/checkpoints/030000/pretrained_model                molmoact2/abl4_wo_stage1_030000
cp1 $O/libero_ablation/abl_latent_only/checkpoints/020000/pretrained_model              molmoact2/abl5_latent_only_020000
# --- π0.5 ---
L=/data2/JM/Code/lerobot/outputs
cp1 $L/pi05_baseline/checkpoints/030000/pretrained_model        pi05/baseline_030000
cp1 $L/pi05_gp_stage1_c10/checkpoints/020000/pretrained_model   pi05/lit_stage1_020000
cp1 $L/pi05_gp_stage2_c10/checkpoints/030000/pretrained_model   pi05/lit_stage2_030000
# --- FastWAM：单文件权重 + 配置 ---
F1=/data2/JM/Code/FastWAM/runs/libero_goal_prior_stage1/stage1_2026-08-21_01-19-56
F2=/data2/JM/Code/FastWAM/runs/libero_goal_prior_stage2/stage2_2026-08-21_02-50-53
mkdir -p $DST/fastwam/lit_stage1 $DST/fastwam/lit_stage2
for s in "$F1:lit_stage1:step_010000.pt" "$F2:lit_stage2:step_030000.pt"; do
  d=${s%%:*}; rest=${s#*:}; rel=${rest%%:*}; f=${rest#*:}
  [ -f "$d/checkpoints/weights/$f" ] && { rsync -a "$d/checkpoints/weights/$f" "$DST/fastwam/$rel/" && rsync -a "$d/config.yaml" "$d/dataset_stats.json" "$DST/fastwam/$rel/" 2>/dev/null; say "OK   fastwam/$rel/$f"; } || say "SKIP fastwam/$rel/$f"
done
# --- ImageWAM ---
I1=/data2/JM/Code/ImageWAM-v2/runs/libero_flux2_klein_4b_goal_prior_stage1/2026-08-17_13-21-30
I2=/data2/JM/Code/ImageWAM-v2/runs/libero_flux2_klein_4b_goal_prior_stage2/2026-08-18_04-16-51
mkdir -p $DST/imagewam/lit_stage1 $DST/imagewam/lit_stage2
[ -f "$I1/checkpoints/weights/step_010000.pt" ] && { rsync -a "$I1/checkpoints/weights/step_010000.pt" "$I1/config.yaml" "$I1/dataset_stats.json" $DST/imagewam/lit_stage1/ 2>/dev/null; say "OK   imagewam/lit_stage1"; } || say "SKIP imagewam/lit_stage1"
[ -f "$I2/checkpoints/weights/step_034720.pt" ] && { rsync -a "$I2/checkpoints/weights/step_034720.pt" "$I2/config.yaml" "$I2/dataset_stats.json" $DST/imagewam/lit_stage2/ 2>/dev/null; say "OK   imagewam/lit_stage2"; } || say "SKIP imagewam/lit_stage2"
say "=== 迁移结束，总量 $(du -sh $DST | cut -f1) ==="
