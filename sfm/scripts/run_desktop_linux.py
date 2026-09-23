#!/usr/bin/env python3
"""Serve the real read-only Open3D window through a private VNC desktop.

All display, authentication, cache and log state stays below DATA_ROOT. Xvfb
uses Linux's abstract Unix socket only, with no /tmp socket or lock file.
VNC and its browser transport bind loopback and require an SSH tunnel.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
DATA = Path("/l/users/anas.khan/cv_802_ass1/sfm")


def main() -> int:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Start in the existing allocation's CPU-only step")
    os.umask(0o077)
    root = DATA / "runtime/desktop"
    sysroot = root / "sysroot"
    logs = DATA / "logs/desktop"
    python = DATA / "envs/open3d-desktop/bin/python"
    display_number = 119
    display = f":{display_number}"
    auth = root / "Xauthority"
    secret_file = root / "vnc-password.txt"
    vnc_auth = root / "vnc-password.bin"
    for directory in (root, logs, root / "run", DATA / "tmp/desktop", DATA / "cache/desktop"):
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "DISPLAY": display, "XAUTHORITY": str(auth),
        "TMPDIR": str(DATA / "tmp/desktop"), "TEMP": str(DATA / "tmp/desktop"),
        "TMP": str(DATA / "tmp/desktop"),
        "PROOT_TMP_DIR": str(DATA / "tmp/desktop"),
        "PYTHONPYCACHEPREFIX": str(DATA / "cache/desktop/pycache"),
        "XDG_CACHE_HOME": str(DATA / "cache/desktop/xdg"),
        "XDG_CONFIG_HOME": str(root / "config"), "XDG_DATA_HOME": str(root / "data"),
        "XDG_RUNTIME_DIR": str(root / "run"), "MPLCONFIGDIR": str(DATA / "cache/desktop/matplotlib"),
        "OPEN3D_DATA_ROOT": str(DATA / "cache/desktop/open3d"),
        "XCOMPOSECACHE": str(DATA / "cache/desktop/xcompose"),
        "LIBGL_ALWAYS_SOFTWARE": "true", "GALLIUM_DRIVER": "llvmpipe",
        "__GLX_VENDOR_LIBRARY_NAME": "mesa", "LP_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
        "XKB_BINDIR": str(sysroot / "usr/bin"),
        "PATH": str(sysroot / "usr/bin") + os.pathsep + env.get("PATH", ""),
        "LD_LIBRARY_PATH": str(sysroot / "usr/lib/x86_64-linux-gnu") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
    })
    # Refuse collision; never kill or replace another display or listener.
    abstract = f"\0/tmp/.X11-unix/X{display_number}"
    probe = socket.socket(socket.AF_UNIX)
    try:
        probe.connect(abstract)
    except OSError:
        pass
    else:
        raise RuntimeError(f"Display {display} already exists; refusing replacement")
    finally:
        probe.close()
    for port in (5906, 8766):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    cookie = secrets.token_hex(16)
    subprocess.run(["xauth", "-f", str(auth), "source", "-"],
                   input=f"add {display} MIT-MAGIC-COOKIE-1 {cookie}\n", text=True, check=True, env=env)
    if not secret_file.exists():
        with secret_file.open("x") as stream:
            stream.write(secrets.token_hex(4) + "\n")
    password = secret_file.read_text().strip()
    x11vnc = sysroot / "usr/bin/x11vnc"
    subprocess.run([str(x11vnc), "-storepasswd", password, str(vnc_auth)],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    children: list[subprocess.Popen] = []
    handles = []

    def start(label: str, command: list[str]) -> subprocess.Popen:
        handle = (logs / f"{label}.log").open("a")
        handles.append(handle)
        child = subprocess.Popen(command, env=env, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        children.append(child)
        print(f"Started {label}: PID {child.pid}; log {logs / (label + '.log')}", flush=True)
        return child

    def stop_signal(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        # PRoot changes paths/UID only for this child (no real root privilege),
        # so Xorg's compiled /usr/bin/xkbcomp and /tmp assumptions stay local.
        xserver = start("xvfb", [str(sysroot / "usr/bin/proot"), "-0",
            "-b", f"{DATA / 'tmp/desktop'}:/tmp",
            "-b", f"{sysroot / 'usr/bin/xkbcomp'}:/usr/bin/xkbcomp",
            str(sysroot / "usr/bin/Xvfb"), display,
            "-screen", "0", "1440x1000x24", "-nolock", "-nolisten", "tcp", "-nolisten", "unix",
            "-auth", str(auth), "-noreset", "-fp", str(sysroot / "usr/share/fonts/X11/misc")])
        for _ in range(100):
            if xserver.poll() is not None:
                raise RuntimeError("Xvfb exited; inspect its data-root log")
            with socket.socket(socket.AF_UNIX) as probe:
                try:
                    probe.connect(abstract)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            raise RuntimeError("Abstract X socket did not become ready")
        # The virtual display must not have placed files in the global /tmp.
        if Path(f"/tmp/.X{display_number}-lock").exists() or Path(f"/tmp/.X11-unix/X{display_number}").exists():
            raise RuntimeError("Display created a file outside DATA_ROOT; refusing to continue")
        start("x11vnc", [str(x11vnc), "-display", display, "-auth", str(auth),
            "-rfbauth", str(vnc_auth), "-listen", "127.0.0.1", "-rfbport", "5906",
            "-forever", "-shared", "-noxdamage", "-noxrecord", "-quiet"])
        start("websockify", [str(python), "-B", "-m", "websockify",
            "--web", str(sysroot / "usr/share/novnc"), "127.0.0.1:8766", "127.0.0.1:5906"])
        gui = start("open3d", [str(python), "-B", str(CODE / "historical_desktop.py"),
            "--experiment", "E10", "--subject", "light", "--width", "1440", "--height", "1000"])
        (root / "session.json").write_text(json.dumps({
            "status": "started_not_yet_validated", "display": display,
            "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
            "children": [child.pid for child in children],
            "browser_url": "http://127.0.0.1:8766/vnc.html", "vnc_port": 5906,
            "password_file": str(secret_file), "software_rendering_requested": True,
        }, indent=2) + "\n")
        print("Real Open3D desktop: loopback HTTP 8766 / VNC 5906. Use SSH forwarding.", flush=True)
        print(f"VNC password is stored privately in {secret_file}", flush=True)
        while gui.poll() is None:
            if any(child.poll() is not None for child in children[:-1]):
                raise RuntimeError("A desktop transport process exited; inspect logs")
            time.sleep(1)
        return gui.returncode
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
