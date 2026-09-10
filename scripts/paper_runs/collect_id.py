#!/usr/bin/env python3
"""LIBERO clean（ID）汇总：从 libero_ablation_eval_id/<arm>/**/eval_info.json 聚合 per_task 成功率 → 每 suite + 平均。"""
import json, glob, os, sys, time, collections
ARM = sys.argv[1]
ROOT = f"/data2/JM/Code/molmoact2/lerobot/outputs/libero_ablation_eval_id/{ARM}"
succ = collections.Counter(); eps = collections.Counter(); seen = set()
for f in glob.glob(os.path.join(ROOT, "**", "eval_info.json"), recursive=True) + glob.glob(os.path.join(ROOT, "**", "live_eval.json"), recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    for t in d.get("per_task", []) or []:
        s = t.get("task_group"); tid = t.get("task_id")
        r = (t.get("metrics", {}) or {}).get("sum_rewards") or (t.get("metrics", {}) or {}).get("successes") or []
        if not s or not r or (s, tid) in seen: continue
        seen.add((s, tid)); succ[s] += sum(1 for x in r if x); eps[s] += len(r)
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
out = {"arm": ARM, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "suites": {}, "episodes": sum(eps.values())}
for s in suites:
    out["suites"][s] = {"succ": succ[s], "eps": eps[s], "pc": (100.0 * succ[s] / eps[s]) if eps[s] else None}
tot_eps = sum(eps.values()); out["avg"] = (100.0 * sum(succ.values()) / tot_eps) if tot_eps else None
out["complete"] = tot_eps >= 2000
json.dump(out, open(f"/data2/JM/abl_{ARM}_id.json", "w"), indent=2)
print(f"[collect_id] {ARM}: " + "  ".join(f"{s.split('_')[1]}={out['suites'][s]['pc']:.1f}({eps[s]})" if eps[s] else f"{s.split('_')[1]}=—" for s in suites) + f"  avg={out['avg']:.2f}" if out["avg"] is not None else f"[collect_id] {ARM}: no results yet", f" eps={tot_eps}/2000")
