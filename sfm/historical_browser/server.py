"""Small dependency-free HTTP server for the read-only E1-E10 WebGL browser."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from .catalog import CatalogError, validate_result
from .combined import CombinedBrowserCatalog


WEB_ROOT = Path(__file__).resolve().parent / "web"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
RESULT_ROUTE = re.compile(r"^/cloud/(E(?:10|[1-9])|MVS|MVSCLEAN|VGGSFM|VGGCLEAN)/(light|dark)\.ply$")
VALIDATE_ROUTE = re.compile(r"^/api/validate/(E(?:10|[1-9])|MVS|MVSCLEAN|VGGSFM|VGGCLEAN)/(light|dark)$")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=False).encode("utf-8")


def make_handler(catalog: CombinedBrowserCatalog) -> type[BaseHTTPRequestHandler]:
    class HistoricalHandler(BaseHTTPRequestHandler):
        server_version = "CV802HistoricalBrowser/1.0"

        def _headers(self, status: int, content_type: str, content_length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'")
            self.end_headers()

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self._headers(status, content_type, len(body))
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, status: int, value: Any) -> None:
            self._send_bytes(status, "application/json; charset=utf-8", _json_bytes(value))

        def _send_error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._route()

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._route()

        def _route(self) -> None:
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/api/catalog":
                    self._send_json(HTTPStatus.OK, catalog.public_payload())
                    return
                validation_match = VALIDATE_ROUTE.fullmatch(path)
                if validation_match:
                    result = catalog.result(
                        validation_match.group(1), validation_match.group(2), require_available=False
                    )
                    report = validate_result(result)
                    status = HTTPStatus.OK if report["valid"] else HTTPStatus.NOT_FOUND
                    self._send_json(status, report)
                    return
                result_match = RESULT_ROUTE.fullmatch(path)
                if result_match:
                    result = catalog.result(result_match.group(1), result_match.group(2))
                    report = validate_result(result)
                    if not report["valid"] or result.ply_path is None:
                        self._send_error(HTTPStatus.NOT_FOUND, report.get("error", "result unavailable"))
                        return
                    size = result.ply_path.stat().st_size
                    self._headers(HTTPStatus.OK, "application/octet-stream", size)
                    if self.command != "HEAD":
                        with result.ply_path.open("rb") as handle:
                            while chunk := handle.read(1024 * 1024):
                                self.wfile.write(chunk)
                    return
                static = STATIC_FILES.get(path)
                if static:
                    name, content_type = static
                    self._send_bytes(HTTPStatus.OK, content_type, (WEB_ROOT / name).read_bytes())
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
            except CatalogError as exc:
                self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            print(f"browser {self.address_string()} - {format % args}")

    return HistoricalHandler


def serve(catalog: CombinedBrowserCatalog, bind: str, port: int) -> None:
    if bind not in {"127.0.0.1", "localhost"}:
        raise CatalogError("the historical browser may bind only to the loopback interface")
    server = ReusableThreadingHTTPServer((bind, port), make_handler(catalog))
    print(f"CV802 E1-E10 + MVS + VGGSfM browser: http://{bind}:{port}")
    print("Read-only: it serves preserved PLY files and never starts reconstruction.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
