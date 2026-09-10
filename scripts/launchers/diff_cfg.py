import json, sys
a_p, b_p, la, lb = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
A, B = json.load(open(a_p)), json.load(open(b_p))

def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out

fa, fb = flat(A), flat(B)
skip = ("output_dir", "job_name", "steps", "save_freq", "wandb", "pretrained_path")
keys = sorted(set(fa) | set(fb))
diffs = []
for k in keys:
    if any(s in k for s in skip):
        continue
    va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
    if va != vb:
        diffs.append((k, va, vb))
print(f"{'field':<48s} {la:<34s} {lb}")
print("-" * 120)
for k, va, vb in diffs:
    print(f"{k:<48s} {str(va):<34s} {vb}")
print(f"\n{len(diffs)} differing fields (output_dir/job_name/steps/save_freq/wandb/pretrained_path excluded)")
