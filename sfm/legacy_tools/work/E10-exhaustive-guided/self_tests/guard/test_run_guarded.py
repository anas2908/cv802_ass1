"""Isolated supervisor tests: no real job, process signal or memory probe runs."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1] / "run_guarded.py"
spec = importlib.util.spec_from_file_location("e10_guard_under_test", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class FakeRunner:
    def __init__(self, pid, events):
        self.pid, self.events, self.code = pid, events, None

    def poll(self):
        return self.code

    def terminate(self):
        self.events.append(("parent_terminate", self.pid))
        self.code = -15

    def wait(self, timeout=None):
        self.events.append(("parent_wait", self.pid, timeout))
        if self.code is None:
            raise AssertionError("Test runner has not reached its scripted exit")
        return self.code


class GuardTests(unittest.TestCase):
    def scenario(self, readings, *, initial_workers=4):
        temporary = tempfile.TemporaryDirectory(prefix="mock_guard_", dir=HERE)
        self.addCleanup(temporary.cleanup)
        work = Path(temporary.name)
        config_path = work / "sfm_refine.json"
        original = {"matching_threads": initial_workers, "matching_batch_size": 128,
                    "resume_matching": True, "max_features": 18000,
                    "matching_pairs": "all_pairs.txt", "guided_pairs": "all_pairs.txt"}
        config_path.write_text(json.dumps(original))
        events, commands, runners, migration_commands = [], [], [], []
        sample = {"index": 0, "free": None}
        alive = set()

        def popen(command, **kwargs):
            commands.append(command)
            runner = FakeRunner(10000 + len(runners), events)
            runners.append(runner)
            alive.add(runner.pid + 100)
            sample["index"] = 0
            return runner

        def quality(parent):
            self.assertEqual(parent, runners[-1].pid)
            current = readings[min(sample["index"], len(readings) - 1)]
            # After fallback the replacement completes under ordinary pressure.
            if len(runners) > 1:
                current = (8e9, 50)
            sample["index"] += 1
            sample["free"] = current[1]
            return {"pid": parent + 100, "resident_bytes": current[0], "cpu_percent": 210.0}

        def fake_kill(pid, sig):
            events.append(("child_signal", pid, sig))
            if pid not in alive:
                raise ProcessLookupError(pid)
            if sig == signal.SIGTERM:
                alive.remove(pid)

        def sleep(seconds):
            if seconds != 5:
                return
            if len(runners) > 1 or sample["index"] >= len(readings):
                runners[-1].code = 0
                alive.discard(runners[-1].pid + 100)
                (work / "status.json").write_text(json.dumps({"complete": True, "subject": "light_shirt"}))

        def migrate(command, **kwargs):
            events.append(("migration", command))
            migration_commands.append(command)
            self.assertIsNotNone(runners[-1].poll())
            self.assertNotIn(runners[-1].pid + 100, alive)
            self.assertTrue(kwargs.get("check"))
            updated = dict(json.loads(config_path.read_text()), matching_threads=2, matching_batch_size=129)
            config_path.write_text(json.dumps(updated))
            return subprocess.CompletedProcess(command, 0)

        with ExitStack() as stack:
            for name, value in (("WORK", work), ("ROOT", work), ("CONFIG", config_path)):
                stack.enter_context(patch.object(guard, name, value))
            stack.enter_context(patch.object(guard.subprocess, "Popen", side_effect=popen))
            stack.enter_context(patch.object(guard.subprocess, "run", side_effect=migrate))
            stack.enter_context(patch.object(guard, "quality_process", side_effect=quality))
            stack.enter_context(patch.object(guard, "free_percent", side_effect=lambda: sample["free"]))
            stack.enter_context(patch.object(guard.time, "sleep", side_effect=sleep))
            stack.enter_context(patch.object(guard.os, "kill", side_effect=fake_kill))
            stack.enter_context(redirect_stdout(io.StringIO()))
            guard.main()
        return {"state": json.loads((work / "supervisor_status.json").read_text()),
                "config": json.loads(config_path.read_text()), "original": original,
                "events": events, "commands": commands, "migrations": migration_commands}

    def test_sustained_pressure_stops_then_migrates_then_restarts_light_only(self):
        result = self.scenario([(8e9, 10)] * 3)
        self.assertEqual(len(result["migrations"]), 1)
        self.assertEqual(result["migrations"][0][-4:], ["--workers", "2", "--batch-size", "129"])
        self.assertEqual([a["workers"] for a in result["state"]["attempts"]], [4, 2])
        self.assertEqual(result["config"], dict(result["original"], matching_threads=2, matching_batch_size=129))
        self.assertTrue(result["state"]["complete"])
        self.assertEqual(result["state"]["stage"], "reconstruction_and_cleanup_complete")
        self.assertEqual(len(result["commands"]), 2)
        for command in result["commands"]:
            self.assertEqual(command[-2:], ["--subject", "light_shirt"])
        events = result["events"]
        terminate = next(i for i, event in enumerate(events) if event[0] == "parent_terminate")
        child_term = next(i for i, event in enumerate(events) if event[0] == "child_signal" and event[2] == signal.SIGTERM)
        waited = next(i for i, event in enumerate(events) if event[0] == "parent_wait")
        migrate = next(i for i, event in enumerate(events) if event[0] == "migration")
        self.assertLess(terminate, child_term)
        self.assertLess(waited, migrate)
        self.assertLess(child_term, migrate)

    def test_urgent_low_free_memory_triggers_immediately(self):
        result = self.scenario([(8e9, 4)])
        self.assertEqual(len(result["migrations"]), 1)

    def test_urgent_rss_can_trigger_without_free_memory_reading(self):
        result = self.scenario([(20.1e9, None)])
        self.assertEqual(len(result["migrations"]), 1)

    def test_success_does_not_fallback(self):
        result = self.scenario([(12e9, 50)] * 2)
        self.assertFalse(result["migrations"])
        self.assertTrue(result["state"]["complete"])
        self.assertEqual(result["config"], result["original"])

    def test_unavailable_percentage_alone_does_not_trigger(self):
        result = self.scenario([(9e9, None)] * 4)
        self.assertFalse(result["migrations"])
        self.assertTrue(result["state"]["complete"])

    def test_pressure_streak_resets(self):
        result = self.scenario([(8e9, 10), (8e9, 50), (8e9, 10), (8e9, 10)])
        self.assertFalse(result["migrations"])

    def test_two_workers_do_not_repeat_fallback(self):
        result = self.scenario([(20.1e9, 3)], initial_workers=2)
        self.assertFalse(result["migrations"])
        self.assertTrue(result["state"]["complete"])

    def test_process_selection_requires_exact_parent_and_light_markers(self):
        rows = "\n".join([
            "8001 99 123 1.0 python run_quality.py light_shirt_quality",
            "8002 42 123 1.0 python run_quality.py black_shirt_crutches_quality",
            "8003 42 123 1.0 python audit_model.py light_shirt_quality",
            "8004 42 456 215.5 python /project/run_quality.py ../datasets/light_shirt_quality",
        ])
        with patch.object(guard.subprocess, "check_output", return_value=rows):
            self.assertEqual(guard.quality_process(42), {"pid": 8004, "resident_bytes": 456 * 1024, "cpu_percent": 215.5})
            self.assertIsNone(guard.quality_process(77))

    def test_unreadable_memory_percentage_returns_none(self):
        with patch.object(guard.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "no memory data", "error")):
            self.assertIsNone(guard.free_percent())

    def test_missing_memory_command_currently_raises(self):
        # Documents the identified robustness gap, rather than running a command.
        with patch.object(guard.subprocess, "run", side_effect=FileNotFoundError("memory_pressure")):
            with self.assertRaises(FileNotFoundError):
                guard.free_percent()

    def test_still_running_child_prevents_migration(self):
        events = []
        runner = FakeRunner(77, events)
        with patch.object(guard.os, "kill", return_value=None), patch.object(guard.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "migration was not attempted"):
                guard.stop(runner, 88)


if __name__ == "__main__":
    unittest.main(verbosity=2)
