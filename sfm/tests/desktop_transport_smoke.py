"""Verify the SSH-forwardable noVNC transport displays the real Open3D desktop."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from playwright.sync_api import sync_playwright, expect

DATA = Path("/l/users/anas.khan/cv_802_ass1/sfm")
SAFE_OUTPUT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--output-id",
        help=(
            "safe run identifier; writes below transport-validation/OUTPUT_ID "
            "so the canonical first receipt is never overwritten"
        ),
    )
    return command


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    output = DATA / "logs/desktop/transport-validation"
    if arguments.output_id:
        if not SAFE_OUTPUT_ID.fullmatch(arguments.output_id) or arguments.output_id in {".", ".."}:
            raise SystemExit(
                "--output-id must be 1-64 letters, digits, dots, underscores or hyphens, "
                "starting with a letter or digit"
            )
        output = output / arguments.output_id
    output.mkdir(parents=True, exist_ok=False)
    password = (DATA / "runtime/desktop/vnc-password.txt").read_text().strip()
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8766/vnc.html?autoconnect=1&resize=scale")
        expect(page.locator("#noVNC_password_input")).to_be_visible(timeout=30000)
        page.fill("#noVNC_password_input", password)
        page.click("#noVNC_credentials_button")
        expect(page.locator("html")).to_have_class(re.compile(r"noVNC_connected"), timeout=30000)
        expect(page.locator("#noVNC_container canvas")).to_be_visible(timeout=30000)
        # Poll actual pixels until a nonempty remote framebuffer arrived.
        page.wait_for_function("""() => {
            const c = document.querySelector('#noVNC_container canvas');
            if (!c || !c.width || !c.height) return false;
            const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            let colored = 0;
            for (let i=0; i<d.length; i+=64) if (d[i]+d[i+1]+d[i+2] > 80) colored++;
            return colored > 100;
        }""", timeout=30000)
        screenshot = output / "real-open3d-desktop.png"
        page.screenshot(path=str(screenshot))
        geometry = page.locator("#noVNC_container canvas").evaluate("c => ({width:c.width,height:c.height})")
        assert not errors, errors
        browser.close()
    record = {
        "status": "complete", "transport": "authenticated VNC over loopback WebSocket",
        "output_id": arguments.output_id,
        "real_remote_framebuffer": geometry, "page_errors": errors,
        "screenshot": {"path": str(screenshot), "bytes": screenshot.stat().st_size,
                       "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest()},
        "limitation": "Tests the transport and rendered pixels; native selection tests are a separate receipt.",
    }
    (output / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
