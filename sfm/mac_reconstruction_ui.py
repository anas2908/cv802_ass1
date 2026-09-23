#!/usr/bin/env python3
"""macOS launcher for fresh SfM experiments.

This UI is deliberately separate from the read-only saved-result browser.
It discovers datasets from ``REPOSITORY/datasets/*/images`` and launches the
portable headless engine in a child process while serving a local browser UI.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import queue
import re
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
VIEWER_WEB = HERE / "historical_browser" / "web"
HISTORICAL_CONFIG = HERE / "configs" / "historical_e1_e10.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPERIMENTS = tuple(f"E{i}" for i in range(1, 11))
SUBJECT_FOLDERS = {"light_shirt": "light_shirt", "dark_shirt": "black_shirt_crutches"}
DEPENDENCIES = {
    "E1": (),
    "E2": ("E1",),
    "E3": ("E1", "E2"),
    "E4": ("E1", "E2", "E3"),
    "E5": ("E1", "E2", "E3"),
    "E6": ("E1", "E2", "E3"),
    "E7": ("E1", "E2", "E3", "E6"),
    "E8": ("E1", "E2", "E3"),
    "E9": ("E1", "E2", "E3", "E8"),
    "E10": ("E1", "E2", "E3"),
}


def discover_datasets(root: Path = REPOSITORY / "datasets") -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not root.is_dir():
        return result
    for child in sorted(root.iterdir()):
        images = child / "images"
        if child.is_dir() and images.is_dir():
            count = sum(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                        for p in images.rglob("*"))
            if count >= 2:
                result[child.name] = images.resolve()
    return result


def validate_selection(dataset: str, experiment: str, datasets: dict[str, Path]) -> None:
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}")
    if experiment == "E10" and dataset != "light_shirt":
        raise ValueError("E10 is intentionally available only for light_shirt")


def experiment_plan(dataset: str, experiment: str, datasets: dict[str, Path]) -> dict:
    validate_selection(dataset, experiment, datasets)
    return {
        "dataset": dataset,
        "images": str(datasets[dataset]),
        "experiment": experiment,
        "prerequisites": list(DEPENDENCIES[experiment]),
        "output_root": str(data_root() / "sfm" / "mac-e1-e10-workspace" /
                           "reconstructions" / SUBJECT_FOLDERS.get(dataset, dataset)),
    }


def data_root() -> Path:
    value = os.environ.get("CV802_DATA_ROOT")
    if not value:
        raise RuntimeError("Set CV802_DATA_ROOT before launching the reconstruction UI")
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_selection(dataset: str, experiment: str) -> int:
    datasets = discover_datasets()
    plan = experiment_plan(dataset, experiment, datasets)
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "last_plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    command = [sys.executable, str(HERE / "mac_recipe_runner.py"), dataset, experiment]
    print("$ " + " ".join(command), flush=True)
    return subprocess.call(command, cwd=HERE, env=dict(os.environ, CV802_ALLOW_NON_SLURM="1"))


def ply_vertex_count(path: Path) -> int:
    """Read the declared vertex count without loading a potentially large PLY."""
    with path.open("rb") as handle:
        header = handle.read(65536)
    end = header.find(b"end_header")
    if end < 0:
        raise ValueError(f"PLY header is incomplete: {path}")
    match = re.search(rb"(?m)^element vertex ([0-9]+)\r?$", header[:end])
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"PLY has no positive vertex count: {path}")
    return int(match.group(1))


def completed_result(dataset: str, experiment: str) -> dict[str, object]:
    """Load and validate the recipe receipt used by the interactive viewer."""
    root = data_root()
    receipt_path = (root / "sfm/mac-e1-e10-workspace/receipts" /
                    f"{dataset}_{experiment}.json")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("dataset") != dataset or receipt.get("experiment") != experiment:
        raise ValueError("The reconstruction receipt does not match the requested run")
    recorded_result = Path(str(receipt.get("result_ply", "")))
    if recorded_result.is_symlink():
        raise ValueError("The reconstructed result may not be a symbolic link")
    result = recorded_result.resolve(strict=True)
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ValueError("The reconstructed result is outside CV802_DATA_ROOT") from error
    if not result.is_file():
        raise ValueError("The reconstructed result is not a regular PLY file")
    catalog = json.loads(HISTORICAL_CONFIG.read_text())
    metadata = next(item for item in catalog["experiments"] if item["id"] == experiment)
    subject = "dark" if dataset == "dark_shirt" else "light"
    return {
        "dataset": dataset,
        "experiment": experiment,
        "subject": subject,
        "path": result,
        "point_count": ply_vertex_count(result),
        "title": metadata["title"],
        "stage_type": metadata["stage_type"],
        "method": metadata["method"],
        "camera_policy": metadata["camera_policy"],
        "display_variant": metadata["display_variant"],
    }


def launch_desktop_ui() -> int:
    if sys.platform != "darwin":
        raise RuntimeError("The reconstruction UI is intended for macOS")
    import tkinter as tk
    from tkinter import messagebox, ttk

    datasets = discover_datasets()
    if not datasets:
        raise RuntimeError("No datasets/*/images folders containing images were found")

    window = tk.Tk()
    window.title("CV802 Mac SfM Reconstruction")
    window.geometry("820x560")
    dataset_var = tk.StringVar(value=next(iter(datasets)))
    experiment_var = tk.StringVar(value="E1")
    status_var = tk.StringVar(value="Ready")
    messages: queue.Queue[str | None] = queue.Queue()
    process: list[subprocess.Popen[str] | None] = [None]

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Dataset").grid(row=0, column=0, sticky="w")
    dataset_box = ttk.Combobox(frame, textvariable=dataset_var,
                               values=list(datasets), state="readonly", width=28)
    dataset_box.grid(row=1, column=0, sticky="ew", padx=(0, 12))
    ttk.Label(frame, text="Experiment").grid(row=0, column=1, sticky="w")
    experiment_box = ttk.Combobox(frame, textvariable=experiment_var,
                                  values=EXPERIMENTS, state="readonly", width=12)
    experiment_box.grid(row=1, column=1, sticky="ew")
    log = tk.Text(frame, wrap="word", height=24)
    log.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(14, 0))
    ttk.Label(frame, textvariable=status_var).grid(row=3, column=0, columnspan=3,
                                                   sticky="w", pady=(10, 0))
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(4, weight=1)

    def refresh_experiments(*_args: object) -> None:
        allowed = list(EXPERIMENTS)
        if dataset_var.get() != "light_shirt":
            allowed.remove("E10")
        experiment_box.configure(values=allowed)
        if experiment_var.get() not in allowed:
            experiment_var.set("E9")

    def pump() -> None:
        while True:
            try:
                line = messages.get_nowait()
            except queue.Empty:
                break
            if line is None:
                status_var.set("Finished")
                run_button.configure(state="normal")
            else:
                log.insert("end", line)
                log.see("end")
        window.after(100, pump)

    def worker(command: list[str]) -> None:
        child = subprocess.Popen(command, cwd=HERE, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process[0] = child
        assert child.stdout is not None
        for line in child.stdout:
            messages.put(line)
        code = child.wait()
        messages.put(f"\nProcess exited with status {code}.\n")
        messages.put(None)
        process[0] = None

    def start() -> None:
        try:
            validate_selection(dataset_var.get(), experiment_var.get(), datasets)
            data_root()
        except (ValueError, RuntimeError) as error:
            messagebox.showerror("Cannot start", str(error))
            return
        run_button.configure(state="disabled")
        status_var.set(f"Running {dataset_var.get()} {experiment_var.get()}")
        command = [sys.executable, str(Path(__file__).resolve()), "run",
                   "--dataset", dataset_var.get(), "--experiment", experiment_var.get()]
        threading.Thread(target=worker, args=(command,), daemon=True).start()

    run_button = ttk.Button(frame, text="Recompute", command=start)
    run_button.grid(row=1, column=2, padx=(12, 0))
    dataset_var.trace_add("write", refresh_experiments)
    refresh_experiments()
    window.protocol("WM_DELETE_WINDOW", lambda: window.destroy()
                    if process[0] is None else messagebox.showwarning(
                        "Reconstruction running", "Wait for the current process to finish."))
    window.after(100, pump)
    window.mainloop()
    return 0


class ReconstructionState:
    """Thread-safe state shared by the local browser UI and one worker."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.label = "Ready"
        self.stage = "Choose a dataset and experiment"
        self.progress = 0
        self.dataset: str | None = None
        self.experiment: str | None = None
        self.result: dict[str, object] | None = None
        self.log: list[str] = []

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "running": self.running,
                "label": self.label,
                "stage": self.stage,
                "progress": self.progress,
                "dataset": self.dataset,
                "experiment": self.experiment,
                "result": dict(self.result) if self.result else None,
                "log": "".join(self.log[-4000:])[-120000:],
            }

    def start(self, dataset: str, experiment: str) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.label = "Running"
            self.stage = f"Starting {dataset} {experiment}"
            self.progress = 1
            self.dataset = dataset
            self.experiment = experiment
            self.result = None
            self.log = []
        threading.Thread(target=self._worker, args=(dataset, experiment), daemon=True).start()
        return True

    def restore_latest(self) -> None:
        """Restore the newest valid local result after the UI is restarted."""
        receipts = data_root() / "sfm/mac-e1-e10-workspace/receipts"
        if not receipts.is_dir():
            return
        for path in sorted(receipts.glob("*.json"), key=lambda item: item.stat().st_mtime,
                           reverse=True):
            try:
                receipt = json.loads(path.read_text())
                dataset = str(receipt["dataset"])
                experiment = str(receipt["experiment"])
                validate_selection(dataset, experiment, discover_datasets())
                result = completed_result(dataset, experiment)
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
            with self.lock:
                self.label = "Previous result"
                self.stage = "Ready to view or recompute"
                self.progress = 100
                self.dataset = dataset
                self.experiment = experiment
                self.result = result
            return

    def _worker(self, dataset: str, experiment: str) -> None:
        command = [sys.executable, str(Path(__file__).resolve()), "run",
                   "--dataset", dataset, "--experiment", experiment]
        try:
            child = subprocess.Popen(command, cwd=HERE, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert child.stdout is not None
            for line in child.stdout:
                with self.lock:
                    if line.startswith("CV802_PROGRESS "):
                        try:
                            update = json.loads(line.removeprefix("CV802_PROGRESS "))
                            value = max(0, min(100, int(update["percent"])))
                            self.progress = max(self.progress, value)
                            self.stage = str(update["stage"])
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            self.log.append(line)
                    else:
                        self.log.append(line)
                        lowered = line.lower()
                        if "extracting features" in lowered:
                            self.progress = max(self.progress, 25)
                            self.stage = "Extracting image features"
                        elif ("sfm: matching" in lowered
                              or ("matching " in lowered and " pairs" in lowered)):
                            self.progress = max(self.progress, 55)
                            self.stage = "Matching features across images"
                        elif "triangulat" in lowered or "estimating cameras" in lowered:
                            self.progress = max(self.progress, 78)
                            self.stage = "Triangulating the 3D points"
            code = child.wait()
            result = completed_result(dataset, experiment) if code == 0 else None
            with self.lock:
                self.log.append(f"\nProcess exited with status {code}.\n")
                self.label = "Completed" if code == 0 else f"Failed (status {code})"
                self.stage = ("Result ready to view" if code == 0
                              else "See the detailed log for the error")
                self.progress = 100 if code == 0 else self.progress
                self.result = result
        except Exception as error:  # Display worker failures in the local UI.
            with self.lock:
                self.log.append(f"\n{type(error).__name__}: {error}\n")
                self.label = "Failed"
                self.stage = "See the detailed log for the error"
        finally:
            with self.lock:
                self.running = False


def render_web_ui(state: ReconstructionState, notice: str = "") -> bytes:
    datasets = discover_datasets()
    status = state.snapshot()
    running = bool(status["running"])
    selected_dataset = status["dataset"]
    selected_experiment = status["experiment"]
    dataset_options = "".join(
        f'<option value="{html.escape(name)}"'
        f'{" selected" if name == selected_dataset else ""}>{html.escape(name)}</option>'
        for name in datasets
    )
    experiment_options = "".join(
        f'<option value="{name}"'
        f'{" selected" if name == selected_experiment else ""}>{name}</option>'
        for name in EXPERIMENTS
    )
    refresh = '<meta http-equiv="refresh" content="2">' if running else ""
    disabled = " disabled" if running else ""
    notice_html = f'<p class="notice">{html.escape(notice)}</p>' if notice else ""
    result = status["result"]
    result_html = ""
    if isinstance(result, dict):
        result_html = (
            '<a class="view-result" href="/viewer?autoload=1">View reconstructed result</a>'
            f'<p class="result-note">{html.escape(str(result["dataset"]))} · '
            f'{html.escape(str(result["experiment"]))} · '
            f'{int(result["point_count"]):,} coloured points</p>'
        )
    progress = int(status["progress"])
    log = str(status["log"])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">{refresh}
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CV802 Mac SfM Reconstruction</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:900px;margin:36px auto;padding:0 20px;background:#f3f6fa;color:#17202a}}
.card{{background:white;border-radius:14px;padding:28px;box-shadow:0 5px 24px #18324f18}}
.controls{{display:flex;gap:14px;align-items:end;flex-wrap:wrap}} label{{display:grid;gap:6px}}
select,button{{font:inherit;padding:10px 12px}} button{{cursor:pointer}}
.status-card{{margin-top:24px;padding:18px;border:1px solid #d9e2ec;border-radius:11px;background:#f8fafc}}
.status-line{{display:flex;justify-content:space-between;gap:20px;font-weight:700}}
.stage{{margin:7px 0 12px;color:#526579}} .progress{{height:14px;overflow:hidden;border-radius:999px;background:#dce5ef}}
.progress>span{{display:block;height:100%;width:{progress}%;border-radius:inherit;background:linear-gradient(90deg,#2476d8,#27b38a);transition:width .4s}}
.view-result{{display:inline-block;margin-top:18px;padding:11px 15px;border-radius:9px;background:#176f55;color:white;text-decoration:none;font-weight:750}}
.result-note{{margin:8px 0 0;color:#526579;font-size:14px}}
details{{margin-top:18px}} summary{{cursor:pointer;color:#526579;font-weight:650}}
pre{{background:#111827;color:#e6edf5;padding:16px;max-height:340px;overflow:auto;white-space:pre-wrap;border-radius:9px;font-size:12px}}
.notice{{color:#a33}}
</style></head><body><main class="card"><h1>CV802 SfM Reconstruction</h1>
<p>Select an included dataset and an E1–E10 experiment. E10 is light-shirt only.</p>
{notice_html}<form class="controls" method="post" action="/run">
<label>Dataset<select id="dataset" name="dataset">{dataset_options}</select></label>
<label>Experiment<select id="experiment" name="experiment">{experiment_options}</select></label>
<button type="submit"{disabled}>Recompute</button></form>
<section class="status-card" aria-live="polite">
<div class="status-line"><span>{html.escape(str(status["label"]))}</span><span>{progress}%</span></div>
<p class="stage">{html.escape(str(status["stage"]))}</p>
<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{progress}"><span></span></div>
{result_html}
</section>
<details><summary>Detailed logs</summary><pre>{html.escape(log) if log else "Logs will appear after reconstruction starts."}</pre></details>
<script>
const dataset = document.getElementById('dataset');
const experiment = document.getElementById('experiment');
function updateExperiments() {{
  const e10 = experiment.querySelector('option[value="E10"]');
  e10.disabled = dataset.value !== 'light_shirt';
  if (e10.disabled && experiment.value === 'E10') experiment.value = 'E9';
}}
dataset.addEventListener('change', updateExperiments); updateExperiments();
</script>
</main></body></html>"""
    return document.encode("utf-8")


def result_catalog(state: ReconstructionState) -> dict[str, object]:
    """Build the one-result catalog consumed by the reusable WebGL viewer."""
    result = state.snapshot()["result"]
    if not isinstance(result, dict):
        raise ValueError("No completed reconstruction is ready to view")
    unavailable = {"available": False, "on_disk": False, "point_count": None,
                   "reason": "This is not the subject selected for the completed run."}
    selected = {"available": True, "on_disk": True,
                "point_count": int(result["point_count"]), "reason": None}
    subjects = {"light": dict(unavailable), "dark": dict(unavailable)}
    subjects[str(result["subject"])] = selected
    return {
        "schema_version": 1,
        "series_note": "Fresh local SfM reconstruction result.",
        "fresh_result": True,
        "experiments": [{
            "id": result["experiment"],
            "title": result["title"],
            "stage_type": result["stage_type"],
            "method": result["method"],
            "camera_policy": result["camera_policy"],
            "display_variant": result["display_variant"],
            "subjects": subjects,
        }],
    }


def result_viewer_page() -> bytes:
    """Reuse the tested point viewer with wording for a fresh reconstruction."""
    page = (VIEWER_WEB / "index.html").read_text()
    replacements = {
        "CV802 Saved 3D Results Browser": "CV802 Reconstructed Result",
        "SfM history": "Recomputed SfM result",
        "Inspect preserved E1–E10, MVS and independent VGGSfM coloured point clouds. This browser is read-only and never starts a reconstruction.":
            "Inspect the coloured point cloud produced by the completed local reconstruction.",
        "Load preserved cloud": "Load reconstructed cloud",
        "Choose an experiment and subject, then load its preserved result.":
            "Loading the reconstructed point cloud…",
    }
    for old, new in replacements.items():
        page = page.replace(old, new)
    page = re.sub(r"\s*<details>.*?</details>", "", page, count=1, flags=re.DOTALL)
    page = page.replace(
        '<body>',
        '<body><a href="/" style="position:fixed;z-index:10;left:12px;top:8px;color:#63d6b5">← Reconstruction</a>',
    )
    return page.encode("utf-8")


def launch_web_ui(port: int = 8770, open_browser: bool = True) -> int:
    if sys.platform != "darwin":
        raise RuntimeError("The reconstruction UI is intended for macOS")
    data_root()
    if not discover_datasets():
        raise RuntimeError("No datasets/*/images folders containing images were found")
    state = ReconstructionState()
    state.restore_latest()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: HTTPStatus, value: object) -> None:
            self.send_bytes(status, "application/json; charset=utf-8",
                            json.dumps(value).encode("utf-8"))

        def send_page(self, notice: str = "", status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(status, "text/html; charset=utf-8", render_web_ui(state, notice))

        def do_GET(self) -> None:  # noqa: N802 - HTTP method name
            route = urlsplit(self.path).path
            if route == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if route in ("/", "/index.html"):
                self.send_page()
                return
            if route == "/viewer":
                try:
                    result_catalog(state)
                    self.send_bytes(HTTPStatus.OK, "text/html; charset=utf-8",
                                    result_viewer_page())
                except ValueError as error:
                    self.send_page(str(error), HTTPStatus.CONFLICT)
                return
            if route == "/api/catalog":
                try:
                    self.send_json(HTTPStatus.OK, result_catalog(state))
                except ValueError as error:
                    self.send_json(HTTPStatus.CONFLICT, {"error": str(error)})
                return
            if route in ("/viewer.js", "/style.css"):
                content_type = ("text/javascript; charset=utf-8" if route.endswith(".js")
                                else "text/css; charset=utf-8")
                self.send_bytes(HTTPStatus.OK, content_type,
                                (VIEWER_WEB / route.removeprefix("/")).read_bytes())
                return
            result = state.snapshot()["result"]
            if isinstance(result, dict):
                expected_route = f'/cloud/{result["experiment"]}/{result["subject"]}.ply'
                if route == expected_route:
                    path = Path(str(result["path"]))
                    size = path.stat().st_size
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with path.open("rb") as handle:
                        while block := handle.read(1024 * 1024):
                            self.wfile.write(block)
                    return
            # Some macOS browsers restore a previous path on the same local
            # port. Bring that tab back to this UI instead of showing a 404.
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - HTTP method name
            if urlsplit(self.path).path != "/run":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                values = parse_qs(self.rfile.read(length).decode("utf-8"))
                dataset = values.get("dataset", [""])[0]
                experiment = values.get("experiment", [""])[0]
                validate_selection(dataset, experiment, discover_datasets())
                notice = ("Reconstruction started." if state.start(dataset, experiment)
                          else "A reconstruction is already running.")
                self.send_page(notice)
            except (ValueError, UnicodeError) as error:
                self.send_page(str(error), HTTPStatus.BAD_REQUEST)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as error:
        raise RuntimeError(
            f"Cannot use port {port}: {error}. Rerun with --port 8771."
        ) from error
    url = f"http://127.0.0.1:{port}/"
    print(f"CV802 reconstruction UI: {url}", flush=True)
    print("Keep this terminal open; press Control-C to stop the UI.", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the reconstruction UI.")
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ui = commands.add_parser("ui", help="open the local macOS browser UI")
    ui.add_argument("--port", type=int, default=8770)
    ui.add_argument("--no-browser", action="store_true")
    commands.add_parser("desktop-ui", help="open the optional legacy Tk window")
    commands.add_parser("list", help="list discovered datasets")
    plan = commands.add_parser("plan", help="print an experiment dependency plan")
    run = commands.add_parser("run", help="run a selected experiment")
    for command in (plan, run):
        command.add_argument("--dataset", required=True)
        command.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    args = parser.parse_args(argv)
    if args.command == "ui":
        return launch_web_ui(args.port, not args.no_browser)
    if args.command == "desktop-ui":
        return launch_desktop_ui()
    datasets = discover_datasets()
    if args.command == "list":
        print(json.dumps({name: str(path) for name, path in datasets.items()}, indent=2))
        return 0
    if args.command == "plan":
        print(json.dumps(experiment_plan(args.dataset, args.experiment, datasets), indent=2))
        return 0
    return run_selection(args.dataset, args.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
