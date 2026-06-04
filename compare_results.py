import json

with open("outputs/polaris_v4_benchmark/v4_final_results.json") as f:
    d = json.load(f)

br = d["baseline_results"]
print("=== BASELINE AGENTS ===")
for k, v in br.items():
    print(f"  {k}: {len(v)} runs")

print("\n=== QWEN 3B BASELINE vs POST-TRAINING ===\n")
base = br.get("Qwen/Qwen2.5-3B-Instruct", [])

for tid in ["environmental_recovery", "balanced_economy"]:
    base_runs = [r for r in base if r["task_id"] == tid]
    print(f"--- {tid} ---")
    print("  BASELINE:")
    for r in base_runs:
        print(f"    seed={r['seed']}: score={r['score']:.4f}, collapsed={r['collapsed']}")
    
    with open("outputs/polaris_v4_benchmark/phase2_3_results.json") as f2:
        p = json.load(f2)
    post = p["post_training_results"].get(tid, [])
    print("  POST-TRAINING:")
    for r in post:
        print(f"    seed={r['seed']}: score={r['score']:.4f}, collapsed={r['collapsed']}")
    
    if base_runs and post:
        b_avg = sum(r["score"] for r in base_runs) / len(base_runs)
        p_avg = sum(r["score"] for r in post) / len(post)
        delta = ((p_avg - b_avg) / max(b_avg, 0.001)) * 100
        print(f"  IMPROVEMENT: {b_avg:.4f} -> {p_avg:.4f} ({delta:+.1f}%)")
    print()
