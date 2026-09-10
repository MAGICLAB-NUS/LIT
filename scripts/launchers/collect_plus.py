#!/usr/bin/env python3
"""Collect live LIBERO-Plus progress for MolmoAct2 baseline vs v3 into one JSON.

Reads the per-shard live_eval.json / eval_info.json written by lerobot_eval and
joins each rollout to its perturbation axis via task_classification.json.
"""
import json, time, os
from pathlib import Path
from collections import defaultdict

OUTS = Path("/data2/JM/Code/molmoact2/lerobot/outputs/libero_eval_plus")
ROOTS = {
    "baseline": OUTS / "baseline_bs224_030000_langfixed" / "libero_plus_full_seed_1000",
    "v3":       OUTS / "v3_030000_langfixed"             / "libero_plus_full_seed_1000",
}
# 2026-08-12 语言 bug 期的两次全量结果，用来在同一批任务上做配对比较
OLD = {
    "baseline": OUTS / "baseline_bs224_030000" / "libero_plus_full_seed_1000",
    "v3":       OUTS / "v3_030000"             / "libero_plus_full_seed_1000",
}
CLS = Path("/data2/JM/Code/molmo_serious/molmoact2-main/third_party/LIBERO-plus"
           "/libero/libero/benchmark/task_classification.json")
OUT = Path("/data2/JM/ma2_plus_progress.json")

AXES = ["Background Textures", "Camera Viewpoints", "Light Conditions", "Objects Layout",
        "Robot Initial States", "Sensor Noise", "Language Instructions"]


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def category_map():
    """(suite, task_id) -> axis / 扰动强度.  task_id is the 0-based index into the suite."""
    cls = json.loads(CLS.read_text(encoding="utf-8"))
    m, totals = {}, defaultdict(int)
    lv, lv_totals = {}, defaultdict(int)
    for suite, entries in cls.items():
        for i, e in enumerate(entries):
            m[(suite, i)] = e["category"]
            totals[e["category"]] += 1
            d = e.get("difficulty_level")
            d = int(d) if d is not None else 0      # 0 = 未标注
            lv[(suite, i)] = d
            lv_totals[d] += 1
    return m, dict(totals), lv, dict(lv_totals)


def collect(root):
    """(suite, task_id) -> bool success, over every shard under root."""
    rec = {}
    if not root.is_dir():
        return rec
    paths = sorted(set(list(root.glob("gpu_*/**/live_eval.json"))
                       + list(root.glob("gpu_*/**/eval_info.json"))))
    for p in paths:
        payload = load(p) or {}
        for task in payload.get("per_task", []) or []:
            suite = str(task.get("task_group") or "")
            if not suite:
                continue
            succ = [bool(v) for v in (task.get("metrics", {}) or {}).get("successes", []) or []]
            if succ:
                rec[(suite, int(task["task_id"]))] = succ[0]
    return rec


def started(root):
    p = root / "run_manifest.json"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None


def main():
    cat, axis_totals, lev, lev_totals = category_map()
    data = {k: collect(r) for k, r in ROOTS.items()}
    old  = {k: collect(r) for k, r in OLD.items()}
    both = set(data["baseline"]) & set(data["v3"])
    # 配对口径：新旧四个 run 都跑过的任务，才拿来算 bug 前后的位移
    paired = both & set(old["baseline"]) & set(old["v3"])

    def block(keys, rec):
        n = len(keys)
        s = sum(1 for k in keys if rec.get(k))
        return {"n": n, "succ": s, "rate": (100.0 * s / n) if n else None}

    axes = []
    for ax in AXES:
        row = {"name": ax, "total": axis_totals.get(ax, 0)}
        for m in ("baseline", "v3"):
            ks = [k for k in data[m] if cat.get(k) == ax]
            row[m] = block(ks, data[m])
        pk = [k for k in paired if cat.get(k) == ax]
        row["paired"] = {
            "n": len(pk),
            "bug_baseline": block(pk, old["baseline"])["rate"],
            "bug_v3":       block(pk, old["v3"])["rate"],
            "new_baseline": block(pk, data["baseline"])["rate"],
            "new_v3":       block(pk, data["v3"])["rate"],
        }
        mk = [k for k in both if cat.get(k) == ax]
        row["matched"] = {
            "n": len(mk),
            "baseline": block(mk, data["baseline"])["rate"],
            "v3": block(mk, data["v3"])["rate"],
        }
        if row["matched"]["baseline"] is not None and row["matched"]["v3"] is not None:
            row["matched"]["delta"] = row["matched"]["v3"] - row["matched"]["baseline"]
        else:
            row["matched"]["delta"] = None
        axes.append(row)

    overall = {m: block(list(data[m]), data[m]) for m in ("baseline", "v3")}
    paired_overall = {
        "n": len(paired),
        "bug_baseline": block(list(paired), old["baseline"])["rate"],
        "bug_v3":       block(list(paired), old["v3"])["rate"],
        "new_baseline": block(list(paired), data["baseline"])["rate"],
        "new_v3":       block(list(paired), data["v3"])["rate"],
    }
    nolang = [k for k in both if cat.get(k) != "Language Instructions"]
    out = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": sum(axis_totals.values()),
        "started": {m: started(r) for m, r in ROOTS.items()},
        "overall": overall,
        "matched_overall": {
            "n": len(both),
            "baseline": block(list(both), data["baseline"])["rate"],
            "v3": block(list(both), data["v3"])["rate"],
        },
        "matched_nolang": {
            "n": len(nolang),
            "baseline": block(nolang, data["baseline"])["rate"],
            "v3": block(nolang, data["v3"])["rate"],
        },
        "axes": axes,
        "levels": [
            {
                "level": L,
                "total": lev_totals.get(L, 0),
                "n": len([k for k in paired if lev.get(k) == L]),
                "bug_baseline": block([k for k in paired if lev.get(k) == L], old["baseline"])["rate"],
                "bug_v3":       block([k for k in paired if lev.get(k) == L], old["v3"])["rate"],
                "new_baseline": block([k for k in paired if lev.get(k) == L], data["baseline"])["rate"],
                "new_v3":       block([k for k in paired if lev.get(k) == L], data["v3"])["rate"],
            }
            for L in sorted(lev_totals)
        ],
        "paired_overall": paired_overall,
        "reference_bugged": {   # 8-12 的 bug 期数字，仅作对照
            "Background Textures": [85.13, 91.91], "Camera Viewpoints": [32.33, 42.96],
            "Light Conditions": [85.11, 86.25], "Objects Layout": [53.51, 66.62],
            "Robot Initial States": [45.03, 52.00], "Sensor Noise": [44.66, 63.71],
            "Language Instructions": [82.82, 81.00], "__overall__": [58.89, 67.28],
        },
    }
    for key in ("matched_overall", "matched_nolang"):
        b, v = out[key]["baseline"], out[key]["v3"]
        out[key]["delta"] = (v - b) if (b is not None and v is not None) else None
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote %s  matched=%d/%d" % (OUT, len(both), out["total_tasks"]))


if __name__ == "__main__":
    main()
