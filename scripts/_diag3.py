"""Verify the relaxed detection logic by stepping through FloorPlan9 and FloorPlan2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from ai2thor.controller import Controller

c = Controller(scene="FloorPlan9", width=300, height=300, renderDepthImage=True)

for scene, target in [("FloorPlan9", "Bowl"), ("FloorPlan2", "Mug")]:
    c.reset(scene=scene)
    try:
        c.step(action="InitialRandomSpawn", randomSeed=51 if scene == "FloorPlan9" else 43,
               forceVisible=False, numPlacementAttempts=5)
    except Exception as e:
        print(f"  InitialRandomSpawn warn: {e}")
    event = c.last_event
    agent = event.metadata["agent"]
    pos = agent["position"]
    heading = float(np.deg2rad(agent["rotation"]["y"]))
    forward = (float(np.sin(heading)), float(np.cos(heading)))
    targets = [o for o in event.metadata["objects"] if o["objectType"] == target]
    print(f"\n=== {scene} target={target} agent@{pos['x']:.2f},{pos['z']:.2f} heading={agent['rotation']['y']:.0f}° ===")
    for t in targets:
        op = t["position"]
        dx = op["x"] - pos["x"]
        dz = op["z"] - pos["z"]
        dist = float(np.hypot(dx, dz))
        cos_a = (forward[0] * dx + forward[1] * dz) / max(dist, 1e-6)
        print(f"  {target} pos=({op['x']:.2f},{op['z']:.2f}) offset=({dx:+.2f},{dz:+.2f}) "
              f"dist={dist:.2f}m  cos_angle={cos_a:+.2f} (need ≥0.707)  "
              f"visible_flag={t.get('visible')}")

    # Rotate around and see what triggers detection
    for k in range(4):
        c.step(action="RotateRight")
        event = c.last_event
        agent = event.metadata["agent"]
        pos = agent["position"]
        heading = float(np.deg2rad(agent["rotation"]["y"]))
        forward = (float(np.sin(heading)), float(np.cos(heading)))
        targets = [o for o in event.metadata["objects"] if o["objectType"] == target]
        for t in targets:
            op = t["position"]
            dx = op["x"] - pos["x"]; dz = op["z"] - pos["z"]
            dist = float(np.hypot(dx, dz))
            cos_a = (forward[0] * dx + forward[1] * dz) / max(dist, 1e-6)
            in_view = t.get("visible") or (dist <= 3.0 and cos_a >= 0.707)
            print(f"  after rotate {k+1} heading={agent['rotation']['y']:.0f}°: "
                  f"dist={dist:.2f} cos={cos_a:+.2f} visible={t.get('visible')} "
                  f"in_view_relaxed={in_view}")

c.stop()
