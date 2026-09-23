"""Real-browser check of E1-E10, MVS and VGGSfM clouds, without CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import threading
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_browser.catalog import HistoricalCatalog
from historical_browser.combined import CombinedBrowserCatalog
from historical_browser.server import ReusableThreadingHTTPServer, make_handler
from playwright.sync_api import sync_playwright, expect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-id",
        help="safe immutable output folder name (default: current UTC timestamp)",
    )
    arguments = parser.parse_args()
    data = Path("/l/users/anas.khan/cv_802_ass1/sfm").resolve(strict=True)
    stamp = arguments.output_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", stamp):
        parser.error("output ID must be a safe path component")
    output = data / "logs" / "browser-checks" / stamp
    output.mkdir(parents=True, exist_ok=False)
    catalog = CombinedBrowserCatalog(HistoricalCatalog())
    server = ReusableThreadingHTTPServer(("127.0.0.1", 0), make_handler(catalog))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    records = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=[
                "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ])
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.evaluate("""localStorage.setItem(
                'cv802.saved-viewer.background.v1',
                JSON.stringify({preset: '__proto__', custom: '#ff00ff'}))
            """)
            page.reload()
            expect(page.locator("#background-preset")).to_have_value("dark")
            expect(page.locator("#background-custom")).to_have_value("#090f1a")
            page.evaluate("localStorage.removeItem('cv802.saved-viewer.background.v1')")
            page.reload()
            expect(page.locator("#experiment option")).to_have_count(14)
            expect(page.locator("#background-preset")).to_have_value("dark")
            expect(page.locator("#background-custom")).to_have_value("#090f1a")
            assert page.is_disabled("#background-custom")
            for number in range(1, 11):
                experiment = f"E{number}"
                page.select_option("#experiment", experiment)
                for subject in (["light", "dark"] if number < 10 else ["light"]):
                    page.check(f"input[name=subject][value={subject}]")
                    page.click("#load")
                    expect(page.locator("#status")).to_contain_text(re.compile(r"coloured points"))
                    expect(page.locator("#load")).to_be_enabled()
                    metrics = page.evaluate("""() => {
                        draw();
                        const data = new Uint8Array(canvas.width * canvas.height * 4);
                        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
                        let visible = 0;
                        for (let i = 0; i < data.length; i += 4) {
                            if (Math.abs(data[i] - 9) + Math.abs(data[i+1] - 15) + Math.abs(data[i+2] - 26) > 10) visible++;
                        }
                        return {points: vertexCount, visiblePixels: visible, glError: gl.getError()};
                    }""")
                    assert metrics["visiblePixels"] > 100, (experiment, subject, metrics)
                    assert metrics["glError"] == 0, metrics
                    expected = catalog.result(experiment, subject).point_count
                    assert metrics["points"] == expected, (experiment, subject, metrics, expected)
                    records.append({"experiment": experiment, "subject": subject, **metrics})
                    if (experiment, subject) in {("E4", "dark"), ("E10", "light")}:
                        page.screenshot(path=str(output / f"{experiment}-{subject}.png"))
            page.select_option("#experiment", "E10")
            assert page.is_disabled("input[name=subject][value=dark]")
            for saved_id in ("MVS", "MVSCLEAN", "VGGSFM", "VGGCLEAN"):
                page.select_option("#experiment", saved_id)
                subjects = [
                    subject
                    for subject in ("light", "dark")
                    if catalog.result(
                        saved_id, subject, require_available=False
                    ).available
                ]
                for subject in subjects:
                    page.check(f"input[name=subject][value={subject}]")
                    page.click("#load")
                    expect(page.locator("#status")).to_contain_text(
                        re.compile(r"coloured points")
                    )
                    metrics = page.evaluate("""() => {
                        draw();
                        const data = new Uint8Array(canvas.width * canvas.height * 4);
                        gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
                        let visible = 0;
                        for (let i = 0; i < data.length; i += 4) {
                            if (Math.abs(data[i] - 9) + Math.abs(data[i+1] - 15) + Math.abs(data[i+2] - 26) > 10) visible++;
                        }
                        return {points: vertexCount, visiblePixels: visible, glError: gl.getError()};
                    }""")
                    expected = catalog.result(saved_id, subject).point_count
                    assert metrics["points"] == expected, (saved_id, subject, metrics, expected)
                    assert metrics["visiblePixels"] > 100, (saved_id, subject, metrics)
                    assert metrics["glError"] == 0, metrics
                    records.append(
                        {"experiment": saved_id, "subject": subject, **metrics}
                    )
                    page.screenshot(path=str(output / f"{saved_id}-{subject}.png"))
                for subject in ("light", "dark"):
                    available = catalog.result(
                        saved_id, subject, require_available=False
                    ).available
                    assert page.is_disabled(
                        f"input[name=subject][value={subject}]"
                    ) == (not available)

            # Background changes are renderer-only: the loaded cloud, colour
            # buffer and point count must survive every preset and custom pick.
            point_state_before = page.evaluate("""() => {
                gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
                window.__browserSmokeColorBuffer = colorBuffer;
                return {
                    points: vertexCount,
                    colorBufferBytes: gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE),
                };
            }""")

            def canvas_clear_pixel() -> list[int]:
                return page.evaluate("""() => {
                    draw();
                    gl.clear(gl.COLOR_BUFFER_BIT);
                    const pixel = new Uint8Array(4);
                    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
                    draw();
                    return Array.from(pixel);
                }""")

            background_records = []
            preset_colours = {
                "dark": ("#090f1a", [9, 15, 26, 255]),
                "black": ("#000000", [0, 0, 0, 255]),
                "white": ("#ffffff", [255, 255, 255, 255]),
                "grey": ("#808080", [128, 128, 128, 255]),
            }
            for preset, (hex_colour, expected_pixel) in preset_colours.items():
                page.select_option("#background-preset", preset)
                expect(page.locator("#background-custom")).to_have_value(hex_colour)
                assert page.is_disabled("#background-custom")
                pixel = canvas_clear_pixel()
                assert pixel == expected_pixel, (preset, pixel, expected_pixel)
                assert page.evaluate("canvas.style.backgroundColor") in {
                    hex_colour,
                    f"rgb({expected_pixel[0]}, {expected_pixel[1]}, {expected_pixel[2]})",
                }
                background_records.append({
                    "preset": preset,
                    "hex": hex_colour,
                    "clear_pixel_rgba": pixel,
                })

            page.select_option("#background-preset", "custom")
            assert not page.is_disabled("#background-custom")
            page.locator("#background-custom").evaluate("""element => {
                element.value = '#2457a6';
                element.dispatchEvent(new Event('input', {bubbles: true}));
            }""")
            expect(page.locator("#background-custom")).to_have_value("#2457a6")
            custom_pixel = canvas_clear_pixel()
            assert custom_pixel == [36, 87, 166, 255], custom_pixel
            point_state_after = page.evaluate("""() => {
                gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
                return {
                    points: vertexCount,
                    colorBufferBytes: gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE),
                    sameColorBuffer: colorBuffer === window.__browserSmokeColorBuffer,
                    glError: gl.getError(),
                };
            }""")
            assert point_state_after == {
                **point_state_before,
                "sameColorBuffer": True,
                "glError": 0,
            }, (point_state_before, point_state_after)
            page.screenshot(path=str(output / "background-custom.png"))

            # Reset, selection changes, another load and a full page refresh do
            # not reset the saved display preference or trigger reconstruction.
            page.click("#fit")
            assert page.evaluate("backgroundHex") == "#2457a6"
            page.select_option("#experiment", "E1")
            page.click("#load")
            expect(page.locator("#status")).to_contain_text("coloured points")
            assert page.evaluate("backgroundHex") == "#2457a6"
            page.reload()
            expect(page.locator("#background-preset")).to_have_value("custom")
            expect(page.locator("#background-custom")).to_have_value("#2457a6")
            assert canvas_clear_pixel() == [36, 87, 166, 255]
            page.click("#load")
            expect(page.locator("#status")).to_contain_text("coloured points")
            before = page.evaluate("yaw")
            page.mouse.move(950, 450)
            page.mouse.down()
            page.mouse.move(1080, 490, steps=10)
            page.mouse.up()
            assert page.evaluate("yaw") != before
            page.click("#fit")
            assert page.evaluate("yaw") == 0.35
            # Exercise the actual wheel handler, beyond BOTH old zoom limits.
            page.locator("#viewer").dispatch_event("wheel", {"deltaY": -1600, "deltaMode": 0})
            close_distance = page.evaluate("distance")
            assert 0.03 <= close_distance < 1.25, close_distance
            page.locator("#viewer").dispatch_event("wheel", {"deltaY": -100000, "deltaMode": 0})
            assert page.evaluate("distance") == 0.03
            page.locator("#viewer").dispatch_event("wheel", {"deltaY": 100000, "deltaMode": 0})
            assert page.evaluate("distance") == 120
            assert page.evaluate("gl.getError()") == 0
            page.click("#fit")
            assert page.evaluate("distance") == 3.1
            def toolbar_layout() -> dict:
                return page.evaluate("""() => {
                const toolbar = document.querySelector('.viewer-toolbar').getBoundingClientRect();
                const shell = document.querySelector('.viewer-shell').getBoundingClientRect();
                const status = document.querySelector('.status').getBoundingClientRect();
                const children = Array.from(document.querySelector('.viewer-toolbar').children)
                    .map(element => element.getBoundingClientRect());
                const overlaps = [];
                for (let left = 0; left < children.length; left += 1) {
                    for (let right = left + 1; right < children.length; right += 1) {
                        const a = children[left], b = children[right];
                        if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 &&
                            Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1) {
                            overlaps.push([left, right]);
                        }
                    }
                }
                return {
                    withinShell: toolbar.left >= shell.left && toolbar.right <= shell.right &&
                        toolbar.top >= shell.top && toolbar.bottom <= shell.bottom,
                    statusSeparated: toolbar.bottom <= status.top,
                    childOverlaps: overlaps,
                    scrollFits: document.querySelector('.viewer-toolbar').scrollWidth <=
                        document.querySelector('.viewer-toolbar').clientWidth,
                };
            }""")

            expected_layout = {
                "withinShell": True,
                "statusSeparated": True,
                "childOverlaps": [],
                "scrollFits": True,
            }
            page.set_viewport_size({"width": 850, "height": 800})
            middle_layout = toolbar_layout()
            assert middle_layout == expected_layout, middle_layout
            page.set_viewport_size({"width": 390, "height": 844})
            mobile_layout = toolbar_layout()
            assert mobile_layout == expected_layout, mobile_layout
            page.screenshot(path=str(output / "mobile-layout.png"))
            assert not errors, errors
            browser.close()
        screenshots = []
        for screenshot in sorted(output.glob("*.png")):
            screenshots.append({
                "path": str(screenshot),
                "bytes": screenshot.stat().st_size,
                "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
            })
        receipt = {"status": "complete", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "renderer": "Chromium SwiftShader software WebGL",
                   "server_bind": "127.0.0.1", "catalog_entries": 14,
                   "results": records, "screenshots": screenshots,
                   "rotation_and_reset_passed": True,
                   "extended_zoom": {"passed": True, "near_distance": 0.03,
                                     "far_distance": 120, "close_test_distance": close_distance,
                                     "reset_distance": 3.1},
                   "background_controls": {
                       "passed": True,
                       "default": {"preset": "dark", "hex": "#090f1a"},
                       "presets": background_records,
                       "custom": {"hex": "#2457a6", "clear_pixel_rgba": custom_pixel},
                       "point_colour_buffer_unchanged": True,
                       "survived_reset_selection_load_and_reload": True,
                       "invalid_saved_preset_fell_back_to_dark": True,
                       "local_storage_key": "cv802.saved-viewer.background.v1",
                       "mid_width_layout": middle_layout,
                       "mobile_layout": mobile_layout,
                   },
                   "page_errors": errors}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"output": str(output), **receipt}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
