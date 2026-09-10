#!/usr/bin/env python3
"""Aggregate LIBERO / LIBERO-Plus evaluation results.

    python3 scripts/aggregate.py <results-root>              # LIBERO-Plus, per axis
    python3 scripts/aggregate.py --libero <results-root>      # LIBERO, per suite
    python3 scripts/aggregate.py <root-a> <root-b> --pair     # paired comparison

The results root is whatever EVAL_ROOT the evaluation wrote to; nested pass/slice
directories are walked, and a task seen more than once keeps its first result.

LIBERO-Plus success rates are reported per perturbation axis. The axes carry unequal
task counts, so the arithmetic mean over axes and the task-weighted rate differ by
roughly two points; the paper reports the arithmetic mean.
"""
import argparse, collections, glob, json, os, sys

AXES = ["Camera Viewpoints", "Sensor Noise", "Light Conditions", "Background Textures",
        "Robot Initial States", "Objects Layout", "Language Instructions"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def classification():
    root = os.environ.get("LIBERO_PLUS_ROOT", "")
    p = os.path.join(root, "libero", "libero", "benchmark", "task_classification.json")
    if not root or not os.path.exists(p):
        sys.exit("error: set LIBERO_PLUS_ROOT so %s can be read" % p)
    cls = json.load(open(p))
    cat, total = {}, collections.Counter()
    for suite, entries in cls.items():
        for i, e in enumerate(entries):
            key = (suite, int(e["id"]) - 1 if "id" in e else i)
            cat[key] = e["category"]
            total[e["category"]] += 1
    return cat, total


def read(root):
    """(suite, task_id) -> 0/1, from every eval_info.json / live_eval.json below root."""
    rec = {}
    for name in ("eval_info.json", "live_eval.json"):
        for f in glob.glob(os.path.join(root, "**", name), recursive=True):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for t in d.get("per_task", []) or []:
                suite = str(t.get("task_group") or "")
                succ = (t.get("metrics", {}) or {}).get("successes", []) or []
                if suite and succ:
                    rec.setdefault((suite, int(t["task_id"])), int(bool(succ[0])))
    return rec


def plus(root, label=None):
    cat, total = classification()
    rec = read(root)
    if not rec:
        sys.exit("no results found under %s" % root)
    ntot = sum(total.values())
    print("%s  %d / %d tasks" % (label or root, len(rec), ntot))
    print("%-22s %6s %6s %8s" % ("axis", "done", "total", "success"))
    vals = []
    for a in AXES:
        ks = [k for k in rec if cat.get(k) == a]
        if not ks:
            print("%-22s %6d %6d %8s" % (a, 0, total[a], "-")); continue
        v = 100.0 * sum(rec[k] for k in ks) / len(ks)
        vals.append(v)
        flag = "" if len(ks) == total[a] else "   (partial — not comparable)"
        print("%-22s %6d %6d %8.2f%s" % (a, len(ks), total[a], v, flag))
    if len(vals) == 7:
        print("%-22s %6s %6s %8.2f   <- reported Overall" % ("mean over axes", "", "", sum(vals) / 7))
    print("%-22s %6d %6s %8.2f" % ("task-weighted", len(rec), "",
                                   100.0 * sum(rec.values()) / len(rec)))
    return rec


def libero(root):
    rec, per = read(root), {}
    if not rec:
        sys.exit("no results found under %s" % root)
    for f in glob.glob(os.path.join(root, "**", "eval_info.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for t in d.get("per_task", []) or []:
            suite = str(t.get("task_group") or "")
            succ = (t.get("metrics", {}) or {}).get("successes", []) or []
            if not suite or not succ:
                continue
            s = per.setdefault(suite, [0, 0])
            s[0] += sum(1 for x in succ if x); s[1] += len(succ)
    print("%-18s %8s %8s %9s" % ("suite", "success", "episodes", "rate"))
    rates, eps = [], 0
    for s in SUITES:
        if s not in per:
            print("%-18s %8s %8s %9s" % (s, "-", "-", "-")); continue
        ok, n = per[s]; eps += n; rates.append(100.0 * ok / n)
        print("%-18s %8d %8d %8.2f%%" % (s, ok, n, 100.0 * ok / n))
    if len(rates) == 4:
        print("%-18s %8s %8d %8.2f%%   <- LIBERO Avg." % ("average", "", eps, sum(rates) / 4))
        if eps != 2000:
            print("note: %d episodes, expected 2000 (50 per task x 40 tasks)" % eps)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--libero", action="store_true", help="in-distribution LIBERO instead of LIBERO-Plus")
    ap.add_argument("--pair", action="store_true", help="compare two roots on their shared tasks")
    a = ap.parse_args()

    if a.libero:
        libero(a.roots[0]); return
    if a.pair:
        if len(a.roots) != 2:
            sys.exit("--pair needs exactly two roots")
        cat, total = classification()
        A, B = read(a.roots[0]), read(a.roots[1])
        shared = sorted(set(A) & set(B))
        print("paired on %d tasks both runs finished\n" % len(shared))
        print("%-22s %6s %8s %8s %8s" % ("axis", "n", "A", "B", "B-A"))
        da = db = []
        ma, mb = [], []
        for ax in AXES:
            ks = [k for k in shared if cat.get(k) == ax]
            if not ks:
                continue
            x = 100.0 * sum(A[k] for k in ks) / len(ks)
            y = 100.0 * sum(B[k] for k in ks) / len(ks)
            ma.append(x); mb.append(y)
            print("%-22s %6d %8.2f %8.2f %+8.2f" % (ax, len(ks), x, y, y - x))
        if len(ma) == 7:
            print("%-22s %6s %8.2f %8.2f %+8.2f" % ("mean over axes", "", sum(ma) / 7, sum(mb) / 7,
                                                    sum(mb) / 7 - sum(ma) / 7))
        return
    for r in a.roots:
        plus(r); print()


if __name__ == "__main__":
    main()
