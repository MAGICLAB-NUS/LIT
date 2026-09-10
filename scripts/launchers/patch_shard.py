import sys, shutil
p = sys.argv[1]; s = open(p).read()
old = 'python - "${EVAL_ROOT}/task_manifest.json" "${SUITES[*]}" "${EVAL_EXCLUDE_DONE_ROOTS:-}" <<\'PYFILTER\''
new = 'python - "${EVAL_ROOT}/task_manifest.json" "${SUITES[*]}" "${EVAL_EXCLUDE_DONE_ROOTS:-}" "${EVAL_TASK_SHARD:-}" <<\'PYFILTER\''
old2 = '''if done:
    tasks = [t for t in tasks if (t["suite"], int(t["task_id"])) not in done]
m["tasks"] = tasks'''
new2 = '''if done:
    tasks = [t for t in tasks if (t["suite"], int(t["task_id"])) not in done]
# 可选：按索引切片 i/n（跨 suite 均衡），用于把长尾 suite 铺到更多实例上
shard = sys.argv[4] if len(sys.argv) > 4 else ""
if shard:
    i, n = (int(x) for x in shard.split("/"))
    tasks = [t for k, t in enumerate(tasks) if k % n == i]
m["tasks"] = tasks'''
old3 = 'print(f"[manifest-filter] suites={sorted(keep_suites)} tasks {before} -> {len(tasks)}"'
new3 = 'print(f"[manifest-filter] suites={sorted(keep_suites)} shard={shard or \'-\'} tasks {before} -> {len(tasks)}"'
for a, b in ((old, new), (old2, new2), (old3, new3)):
    if a not in s: print("ANCHOR MISSING:", a[:60]); sys.exit(2)
if "EVAL_TASK_SHARD" in s: print("already patched"); sys.exit(0)
shutil.copy(p, p + ".pre_shard.bak")
s = s.replace(old, new, 1).replace(old2, new2, 1).replace(old3, new3, 1)
open(p, "w").write(s); print("patched shard support")
