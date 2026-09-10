#!/usr/bin/env python3
"""在 manifest 生成之后、分 shard 之前插入过滤步：
  · 只保留 EVAL_SUITES 里的 suite（修复 4 实例跑同一份清单的 bug）
  · 剔除 EVAL_EXCLUDE_DONE_ROOTS（冒号分隔）下已有结果的 (suite, task_id)，用于续跑
默认（两者都不设）行为不变。"""
import sys, shutil
p = sys.argv[1]; src = open(p).read()
anchor = '  --episodes-per-task "${EPISODES_PER_TASK}"\n'
insert = anchor + r'''
# --- [patch] 按 SUITES 过滤 + 剔除已完成任务（续跑）。默认不改变任何东西。 ---
python - "${EVAL_ROOT}/task_manifest.json" "${SUITES[*]}" "${EVAL_EXCLUDE_DONE_ROOTS:-}" <<'PYFILTER'
import json, sys, glob, os
from pathlib import Path
mp = Path(sys.argv[1]); m = json.loads(mp.read_text(encoding="utf-8"))
keep_suites = set(sys.argv[2].split())
roots = [r for r in sys.argv[3].split(":") if r]
before = len(m["tasks"])
tasks = [t for t in m["tasks"] if t["suite"] in keep_suites]
done = set()
for root in roots:
    for pat in ("**/gpu_*/**/live_eval.json", "**/gpu_*/**/eval_info.json"):
        for f in glob.glob(os.path.join(root, pat), recursive=True):
            try: d = json.load(open(f))
            except Exception: continue
            for t in d.get("per_task", []) or []:
                s = str(t.get("task_group") or "")
                su = (t.get("metrics", {}) or {}).get("successes", []) or []
                if s and su: done.add((s, int(t["task_id"])))
if done:
    tasks = [t for t in tasks if (t["suite"], int(t["task_id"])) not in done]
m["tasks"] = tasks
m["suites"] = [s for s in m["suites"] if s in keep_suites]
m["num_tasks"] = len(tasks)
m["num_rollouts"] = len(tasks) * int(m.get("episodes_per_task", 1))
mp.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
print(f"[manifest-filter] suites={sorted(keep_suites)} tasks {before} -> {len(tasks)}"
      + (f" (excluded {len(done)} already-done pairs from {len(roots)} root(s))" if roots else ""))
PYFILTER
'''
if "[manifest-filter]" in src: print("already patched"); sys.exit(0)
if src.count(anchor) != 1: print("ANCHOR count =", src.count(anchor), "— abort"); sys.exit(2)
shutil.copy(p, p + ".pre_filter.bak")
open(p, "w").write(src.replace(anchor, insert, 1))
print("patched:", p)
