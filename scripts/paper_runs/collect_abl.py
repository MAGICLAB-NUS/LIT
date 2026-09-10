#!/usr/bin/env python3
"""消融臂 vs v3：逐任务配对，按七轴统计。"""
import json, glob, os, sys, time, collections
from pathlib import Path
M="/data2/JM/Code/molmoact2/lerobot/outputs"
ARM=sys.argv[1] if len(sys.argv)>1 else "abl_wo_stage1"
SRC={"base_v3":M+"/libero_eval_plus/baseline_bs224_030000_langfixed/libero_plus_full_seed_1000",
     "ours_v3":M+"/libero_eval_plus/v3_030000_langfixed/libero_plus_full_seed_1000",
     "arm":     M+"/libero_ablation_eval/"+ARM}
CLS="/data2/JM/Code/molmo_serious/molmoact2-main/third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"
OUT=Path("/data2/JM/abl_%s_progress.json"%ARM)
AXES=["Camera Viewpoints","Sensor Noise","Light Conditions","Background Textures",
      "Robot Initial States","Objects Layout","Language Instructions"]
cls=json.load(open(CLS)); cat={}; tot=collections.Counter()
for s,e in cls.items():
    for i,x in enumerate(e): cat[(s,i)]=x["category"]; tot[x["category"]]+=1
def rd(root):
    rec={}
    for pat in ("**/gpu_*/**/live_eval.json","**/gpu_*/**/eval_info.json"):
        for f in glob.glob(os.path.join(root,pat),recursive=True):
            try: d=json.load(open(f))
            except Exception: continue
            for t in d.get("per_task",[]) or []:
                s=str(t.get("task_group") or "")
                if not s: continue
                su=[bool(v) for v in (t.get("metrics",{}) or {}).get("successes",[]) or []]
                if su: rec[(s,int(t["task_id"]))]=int(su[0])
    return rec
D={k:rd(v) for k,v in SRC.items()}
P=set(D["arm"])&set(D["ours_v3"])&set(D["base_v3"])
r=lambda ks,m: 100.0*sum(D[m][k] for k in ks)/len(ks) if ks else None
axes=[]
for a in AXES:
    ks=[k for k in P if cat.get(k)==a]
    axes.append({"name":a,"total":tot[a],"n":len(ks),
                 "base_v3":r(ks,"base_v3"),"ours_v3":r(ks,"ours_v3"),"arm":r(ks,"arm")})
ks=list(P)
OUT.write_text(json.dumps({"arm":ARM,"updated":time.strftime("%Y-%m-%d %H:%M:%S"),
  "total_tasks":sum(tot.values()),"done":len(D["arm"]),"paired":len(P),
  "overall":{"base_v3":r(ks,"base_v3"),"ours_v3":r(ks,"ours_v3"),"arm":r(ks,"arm")},
  "axes":axes},ensure_ascii=False,indent=2),encoding="utf-8")
print("wrote %s  done=%d paired=%d"%(OUT,len(D["arm"]),len(P)))
