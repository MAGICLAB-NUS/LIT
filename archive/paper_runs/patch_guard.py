import sys, shutil
p = sys.argv[1]; s = open(p).read()
old = '''if mismatches:
    raise SystemExit(
        "Refusing to evaluate: checkpoint is not a v2b-compatible semantic-visual Goal-Pose Prior:\\n  "
        + "\\n  ".join(mismatches)
    )'''
new = '''if mismatches and __import__("os").environ.get("EVAL_SKIP_CONFIG_GUARD") != "1":
    raise SystemExit(
        "Refusing to evaluate: checkpoint is not a v2b-compatible semantic-visual Goal-Pose Prior:\\n  "
        + "\\n  ".join(mismatches)
    )
elif mismatches:
    # ablation arms (stagewise / open-visual-path / no-pose-loss) legitimately differ; opt-in skip
    print("[eval] WARNING config guard skipped (EVAL_SKIP_CONFIG_GUARD=1):\\n  " + "\\n  ".join(mismatches))'''
if "EVAL_SKIP_CONFIG_GUARD" in s: print("already patched"); sys.exit(0)
if old not in s: print("ANCHOR NOT FOUND"); sys.exit(2)
s = s.replace(old, new, 1)
# 守卫后的 verified 打印用 .get，避免消融臂 config 缺键时 KeyError
s = s.replace('''    f" mode={cfg['goal_conditioning_mode']}"
    f" tokens={cfg['num_semantic_visual_tokens']}"''', '''    f" mode={cfg.get('goal_conditioning_mode')}"
    f" tokens={cfg.get('num_semantic_visual_tokens')}"''')
shutil.copy(p, p + ".pre_guard.bak"); open(p, "w").write(s); print("patched guard:", p)
