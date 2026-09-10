import json, sys
d = json.load(open(sys.argv[1])); tag = sys.argv[2] if len(sys.argv) > 2 else "arm"
o = d["overall"]
print(f"done={d['done']}/{d['total_tasks']}  paired={d['paired']}  updated={d['updated']}")
print(f"{'':<22}{'n':>6} {'base_v3':>8} {'ours_v3':>8} {tag:>11} {tag+'-base':>10} {tag+'-ours':>10}")
print(f"{'overall':<22}{d['paired']:>6} {o['base_v3']:>8.2f} {o['ours_v3']:>8.2f} {o['arm']:>11.2f} {o['arm']-o['base_v3']:>+10.2f} {o['arm']-o['ours_v3']:>+10.2f}")
for a in d["axes"]:
    if a["n"]:
        print(f"{a['name']:<22}{a['n']:>6} {a['base_v3']:>8.2f} {a['ours_v3']:>8.2f} {a['arm']:>11.2f} {a['arm']-a['base_v3']:>+10.2f} {a['arm']-a['ours_v3']:>+10.2f}")
