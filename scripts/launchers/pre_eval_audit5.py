import json, sys
c = json.load(open(sys.argv[1]))
want = {"enable_goal_pose": True, "mask_image_from_action_expert": True,
        "enable_pose_reconstruction": False, "num_semantic_visual_tokens": 100}
got = {k: c.get(k) for k in want}
print("[pre-eval audit ⑤]", got)
bad = [k for k, v in want.items() if got[k] != v]
print("[pre-eval audit ⑤]", "OK" if not bad else f"MISMATCH: {bad}")
sys.exit(0 if not bad else 3)
