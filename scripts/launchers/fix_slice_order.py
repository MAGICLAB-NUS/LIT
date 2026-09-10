import sys, shutil
p = sys.argv[1]; s = open(p).read()
old = '''if done:
    tasks = [t for t in tasks if (t["suite"], int(t["task_id"])) not in done]
# 可选：按索引切片 i/n（跨 suite 均衡），用于把长尾 suite 铺到更多实例上
shard = sys.argv[4] if len(sys.argv) > 4 else ""
if shard:
    i, n = (int(x) for x in shard.split("/"))
    tasks = [t for k, t in enumerate(tasks) if k % n == i]'''
new = '''# 先切片、后排除：排除必须发生在切片之后，否则并发写入的已完成任务会移除元素、
# 使取模下标整体错位，令部分任务落入无人负责的缝隙（9-9 在 ⑤ 上实测漏 6 条）。
shard = sys.argv[4] if len(sys.argv) > 4 else ""
if shard:
    i, n = (int(x) for x in shard.split("/"))
    tasks = [t for k, t in enumerate(tasks) if k % n == i]
if done:
    tasks = [t for t in tasks if (t["suite"], int(t["task_id"])) not in done]'''
if "先切片、后排除" in s: print("already fixed"); sys.exit(0)
if old not in s: print("ANCHOR MISSING"); sys.exit(2)
shutil.copy(p, p + ".pre_sliceorder.bak")
open(p, "w").write(s.replace(old, new, 1))
print("fixed: slice before exclude")
