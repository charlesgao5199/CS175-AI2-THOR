"""Inspect what Method3 says when the target is visible."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
reasoning = json.loads((ROOT / "results" / "llm_reasoning.json").read_text())

# Find episodes where LLM reasoning mentions seeing the target (not just searching).
for ep in reasoning["episodes"]:
    target = ep["target"]
    seen_steps = []
    for entry in ep["reasoning_log"]:
        r = (entry.get("reasoning") or "")
        # Heuristic: real detection -> reasoning mentions "see", "spot", "found", or target token.
        # The Method3 prompt includes "TARGET seen at grid" when has_target fires; the LLM
        # often echoes this back.
        if any(s in r.lower() for s in ("i see", "i can see", "spotted", "spot", "found")):
            seen_steps.append((entry["step"], entry["action"], r))
    if seen_steps:
        print(f"\n=== {ep['scene']} target={target} success={ep['success']} ===")
        for step, action, r in seen_steps[:8]:
            print(f"  step {step:3d} action={action:14}  {r[:200]}")

# Also, dump entries for FloorPlan9 (Bowl close at start)
print("\n=== FloorPlan9 Bowl — first 10 entries ===")
fp9 = next(ep for ep in reasoning["episodes"] if ep["scene"] == "FloorPlan9")
for entry in fp9["reasoning_log"][:10]:
    print(f"  step {entry['step']:3d} action={entry['action']:14}  {entry['reasoning'][:200]}")
