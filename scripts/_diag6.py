import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "results" / "small_eval.json").read_text())

print("Method3 episodes — distance summary:")
for ep in data["episodes_per_method"]["Method3"]:
    print(f"  {ep['scene']:12} target={ep['target']:11} "
          f"n_targets={ep['n_target_instances']} "
          f"d_init={ep['d_init']:.2f}m d_final={ep['d_final']:.2f}m "
          f"d_taken={ep['d_taken']:.1f}m  stop_issued={ep['stop_issued']}")
