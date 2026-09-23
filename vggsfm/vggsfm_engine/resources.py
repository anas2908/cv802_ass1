"""Sample only an inference process and its descendants; all evidence stays in its attempt."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path


def process_tree(pid: int) -> set[int]:
    """Linux child lists avoid inspecting unrelated users' processes."""
    found: set[int] = set()
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending.extend(int(value) for value in children.split())
        except (OSError, ValueError):
            pass
    return found


def rss_kib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def gpu_memory_mib(pids: set[int]) -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    total = 0
    for line in result.stdout.splitlines():
        try:
            pid, memory = (int(value.strip()) for value in line.split(","))
        except ValueError:
            continue
        if pid in pids:
            total += memory
    return total


class ResourceMonitor:
    """Observed peaks at two-second intervals, not allocator-exact maxima.

    Summed process RSS can double-count shared pages. NVIDIA process accounting
    is limited to this process tree, so another project is never charged to it.
    """

    def __init__(self, pid: int, output: Path, interval: float = 2.0):
        self.pid = pid
        self.output = output
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.summary = {
            "sampling_interval_seconds": interval,
            "samples": 0,
            "peak_process_tree_rss_kib": 0,
            "peak_process_tree_gpu_mib": None,
            "measurement": "sampled process-tree RSS and NVIDIA per-process memory",
            "limitations": "Sampled peaks can miss brief spikes; summed RSS can double-count shared pages.",
            "samples_path": str(output),
        }

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            with self.output.open("x", encoding="utf-8") as stream:
                while not self.stop_event.is_set():
                    pids = process_tree(self.pid)
                    rss = sum(rss_kib(pid) for pid in pids)
                    gpu = gpu_memory_mib(pids)
                    stream.write(json.dumps({"time_unix": time.time(), "pids": sorted(pids), "rss_kib": rss, "gpu_mib": gpu}) + "\n")
                    stream.flush()
                    self.summary["samples"] += 1
                    self.summary["peak_process_tree_rss_kib"] = max(self.summary["peak_process_tree_rss_kib"], rss)
                    if gpu is not None:
                        self.summary["peak_process_tree_gpu_mib"] = max(self.summary["peak_process_tree_gpu_mib"] or 0, gpu)
                    self.stop_event.wait(self.interval)
        except OSError as exc:
            self.summary["sampling_error"] = str(exc)

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=5)
        return dict(self.summary)
