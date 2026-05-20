import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
small = json.loads((ROOT / "results" / "small_eval.json").read_text())
reasoning = json.loads((ROOT / "results" / "llm_reasoning.json").read_text())

# Per-episode: did navigator ever see target?
print("Method3 episodes (reasoning-log signals):")
for ep in reasoning["episodes"]:
    target = ep["target"]
    seen_steps = []
    for entry in ep["reasoning_log"]:
        r = (entry.get("reasoning") or "").lower()
        if "target" in r or target.lower() in r:
            seen_steps.append(entry["step"])
    print(f"  {ep['scene']:12} target={target:11} success={ep['success']!s:5} "
          f"first_target_mention={seen_steps[0] if seen_steps else '-':>4} "
          f"total_steps_mentioning={len(seen_steps)}")

# Per-method summary
print("\nAggregates:")
for name, agg in small["aggregates"].items():
    print(f"  {name:10} success={agg['success']:.2f} spl={agg['spl']:.3f} "
          f"soft_spl={agg['soft_spl']:.3f} avg_steps={agg['avg_steps']:.0f}")

# How many episodes had target objects available?
print("\nMethod3 episodes — target instance count + final distance:")
for ep_compact in small["episodes_per_method"]["Method3"]:
    print(f"  {ep_compact['scene']:12} target={ep_compact['target']:11} "
          f"n_target_instances={ep_compact['n_target_instances']} "
          f"d_init={ep_compact['d_init']:.2f} d_final={ep_compact['d_final']:.2f} "
          f"stop_issued={ep_compact['stop_issued']}")
