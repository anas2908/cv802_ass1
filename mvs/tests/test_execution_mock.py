from __future__ import annotations

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cv802_mvs.commands import StageCommand
from cv802_mvs.runner import _GpuMonitor, execute_command


TEST_TEMP = Path("/l/users/anas.khan/cv_802_ass1/mvs/runtime/unit-tests")


class _Process:
    def __init__(self) -> None:
        self.stdout = io.StringIO("mock colmap output\n")

    def wait(self) -> int:
        return 0


class _Monitor:
    peak_memory = 1234
    peak_utilization = 87

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def __enter__(self) -> "_Monitor":
        return self

    def __exit__(self, *_: object) -> None:
        pass


class ExecutionMockTest(unittest.TestCase):
    def test_argv_is_passed_without_shell_and_output_is_logged_in_data_root(self) -> None:
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            calls = []

            def factory(argv, **kwargs):
                calls.append((argv, kwargs))
                return _Process()

            command = StageCommand("mock", "mock_command", ("colmap", "mock_command", "--x", "a b"))
            log = Path(directory) / "mock.log"
            with patch("cv802_mvs.runner._GpuMonitor", _Monitor):
                metrics = execute_command(
                    command,
                    environment={"PATH": "/usr/bin"},
                    log_path=log,
                    popen_factory=factory,
                )
            self.assertEqual(metrics.return_code, 0)
            self.assertEqual(calls[0][0], list(command.argv))
            self.assertNotIn("shell", calls[0][1])
            self.assertIn("mock colmap output", log.read_text(encoding="utf-8"))

    def test_gpu_monitor_queries_only_slurm_step_gpu(self) -> None:
        environment = {
            "PATH": "/usr/bin",
            "SLURM_STEP_GPUS": "3",
            "SLURM_JOB_GPUS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        monitor = _GpuMonitor(environment, interval_seconds=0.001)
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            monitor._stop.set()
            return subprocess.CompletedProcess(argv, 0, "2048, 73\n", "")

        with patch("cv802_mvs.runner.subprocess.run", side_effect=fake_run):
            monitor._run()

        self.assertEqual(len(calls), 1)
        self.assertIn("--id=3", calls[0][0])
        self.assertNotIn("--id=0", calls[0][0])
        self.assertNotIn("--id=1", calls[0][0])
        self.assertEqual(calls[0][1]["env"], environment)
        self.assertEqual(monitor.peak_memory, 2048)
        self.assertEqual(monitor.peak_utilization, 73)


if __name__ == "__main__":
    unittest.main()
