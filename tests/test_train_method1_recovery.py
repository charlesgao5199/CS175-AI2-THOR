import os
import importlib.util
import signal
import subprocess
import sys
import time
import types
import unittest
from multiprocessing import get_context
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_METHOD1_PATH = REPO_ROOT / "scripts" / "train_method1.py"
spec = importlib.util.spec_from_file_location("train_method1", TRAIN_METHOD1_PATH)
train_method1 = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = train_method1
spec.loader.exec_module(train_method1)
SubprocVecEnv = train_method1.SubprocVecEnv
AI2ThorObjectNavEnv = train_method1.AI2ThorObjectNavEnv


def _pid_running(pid: int) -> bool:
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        state_line = next(
            (line for line in status.read_text(errors="replace").splitlines()
             if line.startswith("State:")),
            "",
        )
        return "\tZ" not in state_line
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _worker_that_leaves_child(conn) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    conn.send(child.pid)
    conn.close()
    time.sleep(60)


@unittest.skipUnless(os.name == "posix", "process-tree cleanup is Linux/WSL-specific")
class SubprocVecEnvRecoveryTests(unittest.TestCase):
    def test_stop_process_terminates_worker_descendants(self) -> None:
        ctx = get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_worker_that_leaves_child, args=(child_conn,))
        proc.start()
        child_conn.close()
        child_pid = int(parent_conn.recv())

        try:
            self.assertTrue(_pid_running(child_pid))

            env = SubprocVecEnv.__new__(SubprocVecEnv)
            env._stop_process(proc)

            deadline = time.time() + 5.0
            while time.time() < deadline and _pid_running(child_pid):
                time.sleep(0.1)

            self.assertFalse(_pid_running(child_pid))
        finally:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            if _pid_running(child_pid):
                _kill_pid(child_pid)


class AI2ThorObjectNavEnvTests(unittest.TestCase):
    def test_controller_uses_configured_server_timeout(self) -> None:
        captured_kwargs = {}

        class FakeController:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                self.last_event = object()

        fake_ai2thor = types.ModuleType("ai2thor")
        fake_controller_module = types.ModuleType("ai2thor.controller")
        fake_controller_module.Controller = FakeController

        previous_ai2thor = sys.modules.get("ai2thor")
        previous_controller = sys.modules.get("ai2thor.controller")
        sys.modules["ai2thor"] = fake_ai2thor
        sys.modules["ai2thor.controller"] = fake_controller_module
        try:
            env = AI2ThorObjectNavEnv(
                scenes=("FloorPlan1",),
                targets=("Mug",),
                server_timeout=12.5,
            )
            env._ensure_controller()
        finally:
            if previous_ai2thor is None:
                sys.modules.pop("ai2thor", None)
            else:
                sys.modules["ai2thor"] = previous_ai2thor
            if previous_controller is None:
                sys.modules.pop("ai2thor.controller", None)
            else:
                sys.modules["ai2thor.controller"] = previous_controller

        self.assertEqual(12.5, captured_kwargs["server_timeout"])


if __name__ == "__main__":
    unittest.main()
