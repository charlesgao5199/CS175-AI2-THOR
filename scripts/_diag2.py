import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
reasoning = json.loads((ROOT / "results" / "llm_reasoning.json").read_text())

# Find episodes where the navigator's own *map-derived* reasoning mentions the target,
# distinct from the LLM monologue. Look for patterns the Method3 code emits.
for ep in reasoning["episodes"]:
    target = ep["target"]
    detection_steps = []
    for entry in ep["reasoning_log"]:
        r = entry.get("reasoning", "") or ""
        # Method3's _describe_surroundings adds "TARGET seen at grid" when has_target fires.
        # Also, the LLM may say "I see the mug at..." which is real.
        # Filter to actual detection: presence of "TARGET seen" OR the target name capitalized.
        if "TARGET" in r and "at grid" in r:
            detection_steps.append(entry["step"])
    # Also check raw response (LLM reasoning could mention it)
    print(f"  {ep['scene']:12} target={target:11} "
          f"detection_via_has_target={len(detection_steps)}  "
          f"success={ep['success']}")

# Sample a few reasoning entries
print("\nSample reasoning from FloorPlan9 (Bowl, d_init=0.60m):")
fp9 = next(ep for ep in reasoning["episodes"] if ep["scene"] == "FloorPlan9")
for entry in fp9["reasoning_log"][:5]:
    print(f"  step {entry['step']:3d} action={entry['action']:14}  reasoning={entry['reasoning'][:120]}")

print("\nSample reasoning from FloorPlan2 (Mug, d_init=1.20m):")
fp2 = next(ep for ep in reasoning["episodes"] if ep["scene"] == "FloorPlan2")
for entry in fp2["reasoning_log"][:5]:
    print(f"  step {entry['step']:3d} action={entry['action']:14}  reasoning={entry['reasoning'][:120]}")
