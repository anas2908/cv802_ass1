#!/usr/bin/env python3
"""Dependency-free browser UI for CIAI MVS and VGGSfM reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import html
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse

from ciai_pipeline import CODE_ROOT, available_datasets, run_pipeline


WEB_ROOT = CODE_ROOT / "sfm" / "historical_browser" / "web"


def ply_point_count(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(65536)
    marker = header.find(b"end_header")
    if marker < 0:
        raise ValueError(f"PLY header is incomplete: {path}")
    match = re.search(rb"(?m)^element vertex ([0-9]+)\r?$", header[:marker])
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"PLY has no positive vertex count: {path}")
    return int(match.group(1))


def discover_results(data_root: Path) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for dataset in available_datasets():
        candidates = (
            (
                "SfM raw",
                data_root / "sfm" / "experiments" / f"ciai-{dataset}-mvs-sfm-v1" / "outputs" / "sparse_colored.ply",
                "CUDA feature extraction, exhaustive matching and incremental sparse mapping",
                "reconstruction",
            ),
            (
                "SfM cleaned",
                data_root / "sfm" / "derived" / f"ciai-{dataset}-sfm-clean-v1" / "sparse_colored_cleaned.ply",
                "Derived geometric cleanup of weak and extreme sparse points; source cameras unchanged",
                "derived_cleanup",
            ),
            (
                "MVS",
                data_root / "mvs" / "experiments" / f"ciai-{dataset}-mvs1600-v1" / "outputs" / "fused.ply",
                "COLMAP CUDA PatchMatch Stereo and depth-map fusion",
                "dense_reconstruction",
            ),
            (
                "VGGSfM",
                data_root / "vggsfm" / "outputs" / f"ciai-{dataset}-vggsfm-v1" / "point_cloud.ply",
                "Pinned official VGGSfM v2 inference",
                "learned_reconstruction",
            ),
        )
        for method, path, description, stage_type in candidates:
            if not path.is_file():
                continue
            try:
                points = ply_point_count(path)
            except (OSError, ValueError):
                continue
            identifier = f"{method.replace(' ', '-')}-{dataset}"
            results[identifier] = {
                "id": identifier,
                "dataset": dataset,
                "method_name": method,
                "description": description,
                "stage_type": stage_type,
                "path": path.resolve(),
                "point_count": points,
            }
    return results


@dataclass
class UIState:
    data_root: Path
    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    dataset: str = ""
    method: str = ""
    percent: int = 0
    stage: str = "Ready"
    error: str | None = None
    result: str | None = None
    logs: list[str] = field(default_factory=list)

    def append(self, line: str) -> None:
        cleaned = line.rstrip()
        if not cleaned:
            return
        with self.lock:
            self.logs.append(cleaned)
            del self.logs[:-1200]
        print(cleaned, flush=True)

    def progress(self, percent: int, stage: str) -> None:
        with self.lock:
            next_percent = min(100, int(percent))
            if next_percent >= self.percent:
                self.percent = next_percent
                self.stage = stage
        print(f"CV802 [{percent:3d}%] {stage}", flush=True)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            payload: dict[str, object] = {
                "running": self.running,
                "dataset": self.dataset,
                "method": self.method,
                "percent": self.percent,
                "stage": self.stage,
                "error": self.error,
                "result": self.result,
                "logs": list(self.logs),
            }
        payload["gpu"] = gpu_status()
        payload["results"] = [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in record.items()
                if key != "path"
            }
            for record in discover_results(self.data_root).values()
        ]
        return payload

    def start(self, dataset: str, method: str) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("A reconstruction is already running")
            if dataset not in available_datasets() or method not in {"sfm", "mvs", "vggsfm"}:
                raise ValueError("Invalid dataset or method")
            self.running = True
            self.dataset = dataset
            self.method = method
            self.percent = 0
            self.stage = "Starting"
            self.error = None
            self.result = None
            self.logs = []

        def worker() -> None:
            try:
                result = run_pipeline(dataset, method, self.data_root, self.progress, self.append)
                with self.lock:
                    self.result = str(result)
            except Exception as error:  # The UI must retain the full failure for diagnosis.
                self.append(f"ERROR: {type(error).__name__}: {error}")
                with self.lock:
                    self.error = str(error)
                    self.stage = "Failed"
            finally:
                with self.lock:
                    self.running = False

        threading.Thread(target=worker, name="cv802-reconstruction", daemon=False).start()


def gpu_status() -> dict[str, object]:
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        command = ["nvidia-smi"]
        if visible and "," not in visible:
            command.append(f"--id={visible}")
        command.extend(
            [
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=4,
        )
        fields = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
        return {
            "available": True,
            "name": fields[0],
            "utilization_percent": int(fields[1]),
            "memory_used_mib": int(fields[2]),
            "memory_total_mib": int(fields[3]),
            "temperature_c": int(fields[4]),
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"available": False}


def main_page(datasets: list[str]) -> bytes:
    options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name.replace("_", " ").title())}</option>'
        for name in datasets
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CV802 CIAI Reconstruction</title><style>
:root{{--bg:#07111f;--panel:#101d2e;--line:#29405c;--text:#eef5ff;--muted:#a9bbcf;--blue:#39a7ff;--red:#ff7081}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px}} h1{{margin:.2rem 0}} .muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}}
label{{display:block;margin:14px 0 6px}} select,button{{width:100%;padding:11px;border-radius:8px;border:1px solid var(--line);font:inherit}}
select{{background:#07111f;color:var(--text)}} button{{margin-top:18px;background:var(--blue);font-weight:700;border:0;cursor:pointer}} button:disabled{{opacity:.45;cursor:not-allowed}}
progress{{width:100%;height:22px;margin:12px 0}} pre{{height:310px;overflow:auto;white-space:pre-wrap;background:#030811;padding:12px;border-radius:8px;color:#cce3ff;font:12px ui-monospace,monospace}}
.gpu{{font-family:ui-monospace,monospace}} .error{{color:var(--red)}} a{{color:#63c0ff}} ul{{padding-left:20px}} @media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<p class="muted">CV802 Project 1 · CIAI A100 workflow</p><h1>MVS and VGGSfM reconstruction</h1>
<p class="muted">Choose one included dataset and one method. All environments, weights, caches and results stay in your own Lustre folder.</p>
<div class="grid"><section class="card">
<label for="dataset">Dataset</label><select id="dataset">{options}</select>
<label for="method">Method</label><select id="method"><option value="sfm">SfM — raw + generic quality cleanup</option><option value="mvs">MVS — SfM + COLMAP PatchMatch</option><option value="vggsfm">VGGSfM — learned reconstruction</option></select>
<button id="start">Start or resume reconstruction</button>
<progress id="progress" max="100" value="0"></progress><div id="stage">Ready</div><p id="error" class="error"></p>
<h3>Completed results</h3><ul id="results"><li class="muted">No CIAI reconstruction completed yet.</li></ul>
</section><section class="card"><h2>Live allocation</h2><div id="gpu" class="gpu">Reading GPU…</div>
<h2>Progress log</h2><pre id="logs">Ready.</pre></section></div></div>
<script>
const $=s=>document.querySelector(s); let prior="";
async function refresh(){{
 const r=await fetch('/api/status'); if(!r.ok)return; const s=await r.json();
 $('#progress').value=s.percent; $('#stage').textContent=`${{s.percent}}% — ${{s.stage}}`; $('#error').textContent=s.error||'';
 $('#start').disabled=s.running; $('#dataset').disabled=s.running; $('#method').disabled=s.running;
 const g=s.gpu; $('#gpu').textContent=g.available?`${{g.name}} · ${{g.utilization_percent}}% GPU · ${{g.memory_used_mib}}/${{g.memory_total_mib}} MiB · ${{g.temperature_c}}°C`:'GPU status unavailable';
 const text=s.logs.join('\\n')||'Ready.'; if(text!==prior){{const p=$('#logs');p.textContent=text;p.scrollTop=p.scrollHeight;prior=text}}
 const list=$('#results'); list.innerHTML=s.results.length?s.results.map(x=>`<li><a href="/viewer?autoload=1&selected=${{encodeURIComponent(x.id)}}" target="_blank">View ${{x.method_name}} · ${{x.dataset}} (${{x.point_count.toLocaleString()}} points)</a></li>`).join(''):'<li class="muted">No CIAI reconstruction completed yet.</li>';
}}
$('#start').onclick=async()=>{{const r=await fetch('/api/start',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{dataset:$('#dataset').value,method:$('#method').value}})}});if(!r.ok)$('#error').textContent=(await r.json()).error;refresh()}};
refresh();setInterval(refresh,2000);
</script></body></html>"""
    return page.encode()


def catalog(data_root: Path) -> dict[str, object]:
    experiments = []
    for record in discover_results(data_root).values():
        dataset = str(record["dataset"])
        subject = "dark" if "dark" in dataset.lower() else "light"
        missing = {"available": False, "on_disk": False, "point_count": 0, "reason": "Different dataset"}
        available = {"available": True, "on_disk": True, "point_count": record["point_count"], "reason": ""}
        experiments.append(
            {
                "id": record["id"],
                "title": f'{record["method_name"]} · {dataset}',
                "method": record["description"],
                "camera_policy": "Estimated from this dataset",
                "display_variant": "raw coloured output",
                "stage_type": record["stage_type"],
                "subjects": {"light": available if subject == "light" else missing, "dark": available if subject == "dark" else missing},
            }
        )
    return {"fresh_result": True, "experiments": experiments}


def build_handler(state: UIState, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CV802-CIAI/1.0"

        def log_message(self, format: str, *args: object) -> None:
            return

        def authenticated(self) -> bool:
            query = parse_qs(urlparse(self.path).query)
            if query.get("token", [""])[0] == token:
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Set-Cookie", f"cv802_token={token}; HttpOnly; SameSite=Strict; Path=/")
                self.send_header("Location", "/")
                self.end_headers()
                self._authentication_response_sent = True
                return False
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            return cookie.get("cv802_token") is not None and cookie["cv802_token"].value == token

        def json_response(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def bytes_response(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def file_response(self, path: Path, content_type: str) -> None:
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

        def do_GET(self) -> None:
            if not self.authenticated():
                if not getattr(self, "_authentication_response_sent", False):
                    self.send_error(HTTPStatus.FORBIDDEN, "Open the tokenized URL printed by the launcher")
                return
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.bytes_response(main_page(available_datasets()), "text/html; charset=utf-8")
            elif parsed.path == "/api/status":
                self.json_response(state.snapshot())
            elif parsed.path == "/api/catalog":
                self.json_response(catalog(state.data_root))
            elif parsed.path == "/viewer":
                self.bytes_response((WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/viewer.js":
                self.bytes_response((WEB_ROOT / "viewer.js").read_bytes(), "text/javascript; charset=utf-8")
            elif parsed.path == "/style.css":
                self.bytes_response((WEB_ROOT / "style.css").read_bytes(), "text/css; charset=utf-8")
            elif parsed.path.startswith("/cloud/") and parsed.path.endswith(".ply"):
                parts = unquote(parsed.path).split("/")
                identifier = parts[2] if len(parts) == 4 else ""
                record = discover_results(state.data_root).get(identifier)
                if record is None:
                    self.send_error(404)
                    return
                cloud = Path(record["path"])
                self.file_response(cloud, "application/octet-stream")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if not self.authenticated():
                if not getattr(self, "_authentication_response_sent", False):
                    self.send_error(HTTPStatus.FORBIDDEN)
                return
            if urlparse(self.path).path != "/api/start":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                state.start(str(payload.get("dataset", "")), str(payload.get("method", "")))
                self.json_response({"started": True}, 202)
            except (ValueError, RuntimeError, json.JSONDecodeError) as error:
                self.json_response({"error": str(error)}, 409)

    return Handler


def gpu_monitor(state: UIState) -> None:
    while True:
        time.sleep(30)
        with state.lock:
            running = state.running
        if running:
            gpu = gpu_status()
            if gpu.get("available"):
                print(
                    f"GPU {gpu['utilization_percent']}% · "
                    f"{gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB · "
                    f"{gpu['temperature_c']}°C",
                    flush=True,
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    if not args.data_root.is_absolute() or len(args.token) < 24:
        parser.error("an absolute data root and a strong token are required")
    state = UIState(args.data_root.resolve())
    threading.Thread(target=gpu_monitor, args=(state,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state, args.token))
    print(f"CV802 CIAI UI listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping UI.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
