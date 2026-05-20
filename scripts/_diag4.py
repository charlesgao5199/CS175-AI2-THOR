"""Run a single Method2 episode on FloorPlan9 with detection-trace prints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from ai2thor.controller import Controller

from method2 import Method2Navigator
from shared.interfaces import Action

# Reuse the eval's _event_to_obs verbatim
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_small_eval import _event_to_obs, ACTION_TO_THOR

c = Controller(scene="FloorPlan9", width=300, height=300, renderDepthImage=True)
c.reset(scene="FloorPlan9")
try:
    c.step(action="InitialRandomSpawn", randomSeed=50, forceVisible=False, numPlacementAttempts=5)
except Exception as e:
    print(f"InitialRandomSpawn warn: {e}")

event = c.last_event
start_pos = dict(event.metadata["agent"]["position"])
agent = event.metadata["agent"]
print(f"Start: pos={start_pos} heading={agent['rotation']['y']}°")
print(f"Bowl positions: {[(o['position']['x'], o['position']['z']) for o in event.metadata['objects'] if o['objectType']=='Bowl']}")

nav = Method2Navigator()
nav.reset("Bowl")

for step in range(30):
    obs = _event_to_obs(event, "Bowl", start_pos)
    vis_targets = [v for v in obs.visible_objects if v["name"] == "Bowl"]
    has_t = nav.builder.update(obs).has_target("Bowl") if step == 0 else None
    if vis_targets:
        v = vis_targets[0]
        print(f"  step {step:2d} pre-act: visible_Bowl=YES dx={v['dx']:+.2f} dz={v['dz']:+.2f}")
    action = nav.act(obs)
    smap = nav.builder.map
    cell = smap.has_target("Bowl")
    print(f"  step {step:2d} action={action.name:14} has_target={cell} reasoning={nav.last_reasoning[:80]}")
    if action == Action.STOP:
        print(f"  STOP issued.")
        break
    event = c.step(action=ACTION_TO_THOR[action])

c.stop()
