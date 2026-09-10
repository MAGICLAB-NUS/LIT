#!/usr/bin/env python3
"""Build docs/index.html. Every table number comes from the v10 PDF, then is verified back."""
import re, io, html
from pypdf import PdfReader
import sys
PDF = sys.argv[1] if len(sys.argv) > 1 else "paper.pdf"   # the submission PDF
NUM = re.compile(r'[+−\-]?\d+\.\d{2}')
P = [pg.extract_text() or "" for pg in PdfReader(PDF).pages]

def nums_between(page, start, end_pats):
    t = P[page]; s = t.find(start); seg = t[s:]
    for ep in end_pats:
        e = re.search(ep, seg)
        if e: seg = seg[:e.start()]
    return [x.replace("−","-") for x in NUM.findall(seg)]

# ---- Table I (LIBERO) : 8 rows x 5
t1 = nums_between(4, "TABLE I", [r"TABLE II", r"Fig\. 3"])
assert len(t1) >= 40, len(t1); t1 = t1[:40]
# ---- Table II (LIBERO-Plus) : 8 rows x 12
t2 = nums_between(5, "TABLE II", [r"Fig\. 3", r"Real-robot"])
assert len(t2) == 96, len(t2)
# ---- Table III (ablation) : 8 rows x 9
t3 = nums_between(7, "TABLE III", [r"Vanilla staged training\.", r"0 2k 4k"])
assert len(t3) >= 72, len(t3); t3 = t3[:72]

ARCH = ["π<sub>0.5</sub>", "MolmoAct2", "FAST-WAM", "ImageWAM"]
KIND = ["VLA", "VLA", "WAM", "WAM"]
SUITES = ["Spatial", "Object", "Goal", "Long", "Avg."]
AXES = ["Camera Viewpoints","Sensor Noise","Lighting Conditions","Background Textures",
        "Robot Initial States","Object Layout","Language Instructions","Overall"]
ABL = ["MolmoAct2","Vanilla staged training","LIT w/o Stage 1 &amp; pose supervision","LIT w/o Stage 1",
       "LIT w/o pose supervision","Baseline w/ pose supervision","LIT w/ direct visual access","LIT"]

def f(x): return x
def cell(v, bold=False, cls=""):
    c = ' class="%s"' % cls if cls else ""
    return "<td%s>%s</td>" % (c, ("<b>%s</b>" % v) if bold else v)
def dcell(v):
    cls = "d-up" if v.startswith("+") else "d-dn"
    return '<td class="num %s">%s</td>' % (cls, v.replace("-", "&minus;"))

# Table I
rows = []
for a in range(4):
    b = t1[a*10:a*10+5]; l = t1[a*10+5:a*10+10]
    rows.append('<tr class="sep"><td class="l" rowspan="2">%s <span class="kind">%s</span></td><td class="l">Baseline</td>%s</tr>'
        % (ARCH[a], KIND[a], "".join(cell(b[i], float(b[i])>float(l[i]), "num") for i in range(5))))
    rows.append('<tr class="lit"><td class="l">LIT</td>%s</tr>'
        % "".join(cell(l[i], float(l[i])>float(b[i]), "num") for i in range(5)))
TABLE1 = ('<table class="t1"><thead><tr><th class="l">Model</th><th class="l">Variant</th>%s</tr></thead><tbody>%s</tbody></table>'
    % ("".join("<th>%s</th>"%s for s in SUITES), "".join(rows)))

# Table II
head = ('<thead><tr><th class="l" rowspan="2">Perturbation</th>%s</tr><tr>%s</tr></thead>'
    % ("".join('<th class="grp" colspan="3">%s <span class="kind">(%s)</span></th>' % (ARCH[a], KIND[a]) for a in range(4)),
       "".join('<th class="axis">Base</th><th class="axis">LIT</th><th class="axis">&Delta;</th>' for _ in range(4))))
rows = []
for r in range(8):
    v = t2[r*12:(r+1)*12]; tds = []
    for a in range(4):
        b, l, d = v[a*3], v[a*3+1], v[a*3+2]
        tds.append(cell(b, float(b)>float(l), "num")); tds.append(cell(l, float(l)>float(b), "num")); tds.append(dcell(d))
    rows.append('<tr class="%s"><td class="l">%s</td>%s</tr>' % ("lit sep" if r==7 else "", AXES[r], "".join(tds)))
TABLE2 = '<table class="t2">%s<tbody>%s</tbody></table>' % (head, "".join(rows))

# Table III
head = ('<thead><tr><th class="l" rowspan="2">Variant</th><th class="grp">ID</th><th class="grp" colspan="8">OOD: LIBERO-Plus</th></tr>'
        '<tr><th class="axis">LIBERO Avg.</th>%s<th class="axis">Overall</th></tr></thead>'
        % "".join('<th class="axis">%s</th>' % s for s in ["Camera","Noise","Lighting","Backgr.","Robot","Layout","Lang."]))
rows = []
for r in range(8):
    v = t3[r*9:(r+1)*9]
    cls = "sep" if r in (0,1,7) else ""
    if r == 7: cls += " lit"
    rows.append('<tr class="%s"><td class="l">%s</td>%s</tr>' % (cls, ABL[r], "".join(cell(x, False, "num") for x in v)))
TABLE3 = '<table class="t3">%s<tbody>%s</tbody></table>' % (head, "".join(rows))

# ---- Fig. 1 caption, verbatim if it can be located
FIG1CAP = ("<b>Fig. 1. Training paradigms and generalization performance.</b> Standard training conditions the "
           "action expert on visual representations (left). LIT (right) first learns a spatial-goal-conditioned "
           "action prior without images to avoid relying on visual correlations during pre-training (Stage 1), "
           "then introduces visual conditioning through a pose-supervised latent interface (Stage 2). Across four "
           "VLA and WAM architectures, LIT improves out-of-distribution success rates while preserving or "
           "improving in-distribution success rates (center).")
print("Fig.1 caption:", re.sub(r"<[^>]+>", "", FIG1CAP)[:160], "...")

# ---- Overall-row deltas from Table II drive the bar chart and the headline stat
ov = t2[7*12:8*12]
deltas = [float(ov[a*3+2]) for a in range(4)]
bases  = [ov[a*3] for a in range(4)]; lits = [ov[a*3+1] for a in range(4)]
mx = max(deltas)
DELTA_CHART = "".join(
    '<div class="bar"><span class="nm">%s<span class="kind">%s</span></span>'
    '<span class="tr"><span class="fl" data-w="%.1f"></span></span>'
    '<span class="v">%s<small>%s &rarr; %s</small></span></div>'
    % (ARCH[a], KIND[a], 100.0*deltas[a]/mx, ov[a*3+2], bases[a], lits[a]) for a in range(4))
MAXGAIN = "%.2f" % mx
print("Table II overall deltas:", [ov[a*3+2] for a in range(4)], "-> headline +%s" % MAXGAIN)

tpl = io.open(sys.argv[2] if len(sys.argv) > 2 else "template.html", encoding="utf-8").read()
out = (tpl.replace("{{TABLE1}}", TABLE1).replace("{{TABLE2}}", TABLE2)
          .replace("{{TABLE3}}", TABLE3).replace("{{FIG1CAP}}", FIG1CAP)
          .replace("{{DELTA_CHART}}", DELTA_CHART).replace("{{MAXGAIN}}", MAXGAIN))
io.open("index.html", "w", encoding="utf-8").write(out)

# ---- verify: numbers in HTML tables == numbers from PDF, in order
def html_nums(frag):
    txt = re.sub(r"<[^>]+>", " ", frag).replace("&minus;", "-")
    return NUM.findall(txt)
for name, frag, ref in (("Table I", TABLE1, t1), ("Table II", TABLE2, t2), ("Table III", TABLE3, t3)):
    got = html_nums(frag)
    ok = got == ref
    print("%-9s PDF %3d 个数 | HTML %3d 个数 | %s" % (name, len(ref), len(got), "逐个一致 ✓" if ok else "★ 不一致"))
    if not ok:
        for i,(a,b) in enumerate(zip(ref,got)):
            if a!=b: print("   第%d个: PDF %s vs HTML %s" % (i,a,b)); break
print("index.html:", len(out), "bytes")
