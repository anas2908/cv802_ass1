"""Checks for attributing memory only to the requested inference process."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vggsfm_engine.resources import ResourceMonitor, gpu_memory_mib, process_tree


class ResourceTests(unittest.TestCase):
    def test_nvidia_rows_exclude_unrelated_processes(self):
        result = subprocess.CompletedProcess([], 0, "12, 4096\n34, 8192\n56, 128\n12, N/A\n", "")
        with mock.patch("vggsfm_engine.resources.subprocess.run", return_value=result):
            self.assertEqual(gpu_memory_mib({12, 56}), 4224)

    def test_missing_nvidia_is_unavailable_not_zero(self):
        with mock.patch("vggsfm_engine.resources.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(gpu_memory_mib({12}))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc measurement")
    def test_monitor_writes_under_explicit_attempt_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resources.jsonl"
            monitor = ResourceMonitor(os.getpid(), output, interval=0.01)
            with mock.patch("vggsfm_engine.resources.gpu_memory_mib", return_value=123):
                monitor.start()
                # Wait for a sampled measurement, bounded by the thread timeout.
                import time
                deadline = time.monotonic() + 2
                while not monitor.summary["samples"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                result = monitor.finish()
            self.assertGreater(result["samples"], 0)
            self.assertGreater(result["peak_process_tree_rss_kib"], 0)
            self.assertEqual(result["peak_process_tree_gpu_mib"], 123)
            self.assertTrue(output.read_text().strip())
            self.assertFalse(monitor.thread.is_alive())
            self.assertIn(os.getpid(), process_tree(os.getpid()))
