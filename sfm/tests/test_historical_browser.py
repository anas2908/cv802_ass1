"""Tests for the read-only E1-E10 selector and HTTP point-cloud browser."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from http.server import HTTPServer
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from historical_browser.catalog import (
    CATALOG_PATH,
    CatalogError,
    HistoricalCatalog,
    read_ply_header,
    validate_result,
)
from historical_browser.combined import CombinedBrowserCatalog
from historical_browser.cli import main
from historical_browser.server import ReusableThreadingHTTPServer, make_handler, serve
from historical_desktop import DesktopSelection


def catalog_payload() -> dict:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_catalog(directory: Path, payload: dict) -> Path:
    path = directory / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_desktop_views(directory: Path, count: int = 2) -> Path:
    path = directory / "desktop_views.json"
    payload = {
        "schema_version": 1,
        "views": [
            {
                "id": "MVS",
                "title": "Fixture dense fusion",
                "stage_type": "dense_reconstruction",
                "method": "mvs",
                "subjects": {
                    "light": {
                        "available": True,
                        "point_count": count,
                        "relative_ply": "mvs/experiment/outputs/fused.ply",
                    },
                    "dark": {
                        "available": False,
                        "point_count": None,
                        "relative_ply": None,
                        "reason": "fixture dark MVS is not complete",
                    },
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_binary_ply(path: Path, count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertices = b"".join(
        struct.pack("<fffBBB", float(index), 0.0, 1.0, 10, 20, 30)
        for index in range(count)
    )
    path.write_bytes(header + vertices)


class HistoricalCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_catalog(self, payload: dict | None = None) -> HistoricalCatalog:
        data = deepcopy(payload or catalog_payload())
        data["reconstruction_root"] = "historical/reconstructions"
        return HistoricalCatalog(self.root, write_catalog(self.root, data))

    def test_catalog_has_ordered_e1_to_e10_and_correct_stage_lineage(self) -> None:
        catalog = self.make_catalog()
        desktop = DesktopSelection(catalog)
        self.assertEqual([item["id"] for item in catalog.experiments()], [f"E{i}" for i in range(1, 11)])
        self.assertEqual(
            desktop.experiment_ids,
            (*tuple(f"E{i}" for i in range(1, 11)), "MVS", "MVSCLEAN", "VGGSFM", "VGGCLEAN"),
        )
        self.assertEqual(desktop.available_subjects("E10"), ("light",))
        self.assertEqual(desktop.available_subjects("MVS"), ("light", "dark"))
        self.assertEqual(desktop.available_subjects("MVSCLEAN"), ("dark",))
        self.assertEqual(desktop.available_subjects("VGGSFM"), ("light", "dark"))
        self.assertEqual(desktop.available_subjects("VGGCLEAN"), ("light", "dark"))
        self.assertIn("[cleanup]", desktop.experiment_label("E4"))
        self.assertIn("[reconstruction]", desktop.experiment_label("E10"))
        self.assertIn("[dense MVS]", desktop.experiment_label("MVS"))
        self.assertIn("[cleanup]", desktop.experiment_label("MVSCLEAN"))
        self.assertIn("[learned SfM]", desktop.experiment_label("VGGSFM"))
        self.assertIn("[cleanup]", desktop.experiment_label("VGGCLEAN"))
        cleanups = {
            item["id"] for item in catalog.experiments() if item["stage_type"] == "derived_cleanup"
        }
        self.assertEqual(cleanups, {"E2", "E4", "E5", "E7", "E9"})
        self.assertFalse(catalog.result("E10", "dark", require_available=False).available)
        with self.assertRaisesRegex(CatalogError, "light shirt only"):
            catalog.result("E10", "dark")
        self.assertEqual(desktop.experiment("MVS")["subjects"]["dark"]["point_count"], 616827)
        self.assertEqual(
            desktop.experiment("VGGSFM")["subjects"]["dark"]["point_count"],
            187308,
        )
        self.assertEqual(
            desktop.experiment("VGGCLEAN")["subjects"]["dark"]["point_count"],
            62889,
        )

    def test_resolves_documented_display_path_below_data_root(self) -> None:
        catalog = self.make_catalog()
        result = catalog.result("e4", "light")
        self.assertEqual(result.point_count, 19519)
        self.assertEqual(
            result.ply_path,
            self.root
            / "historical/reconstructions/experiments/light_person_mask_cleanup/mask_consensus_90/subject.ply",
        )

    def test_catalog_rejects_parent_escape(self) -> None:
        payload = catalog_payload()
        payload["experiments"][0]["subjects"]["light"]["relative_ply"] = "../escape.ply"
        with self.assertRaisesRegex(CatalogError, "without '..'"):
            self.make_catalog(payload)

    def test_result_rejects_symlink_escape(self) -> None:
        catalog = self.make_catalog()
        outside = self.root / "outside"
        outside.mkdir()
        reconstruction_root = self.root / "historical/reconstructions"
        reconstruction_root.mkdir(parents=True)
        (reconstruction_root / "light_shirt").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(CatalogError, "escapes"):
            catalog.result("E1", "light")

    def test_ply_header_and_documented_count_validation(self) -> None:
        payload = catalog_payload()
        payload["experiments"][0]["subjects"]["light"]["point_count"] = 2
        catalog = self.make_catalog(payload)
        result = catalog.result("E1", "light")
        write_binary_ply(result.ply_path, 2)
        self.assertEqual(read_ply_header(result.ply_path)["vertex_count"], 2)
        self.assertTrue(validate_result(result)["valid"])
        selected = DesktopSelection(catalog).resolve("E1", "light")
        self.assertEqual(selected["result"], result)
        self.assertTrue(selected["validation"]["valid"])

        desktop = DesktopSelection(catalog, write_desktop_views(self.root, 2))
        fused = self.root / "mvs/experiment/outputs/fused.ply"
        write_binary_ply(fused, 2)
        dense = desktop.resolve("MVS", "light")
        self.assertEqual(dense["result"].ply_path, fused)
        self.assertEqual(dense["validation"]["ply"]["vertex_count"], 2)

        payload["experiments"][0]["subjects"]["light"]["point_count"] = 3
        mismatched = self.make_catalog(payload).result("E1", "light")
        report = validate_result(mismatched)
        self.assertFalse(report["valid"])
        self.assertIn("does not match documented", report["error"])

    def test_vggsfm_saved_view_requires_complete_hash_bound_manifest(self) -> None:
        catalog = self.make_catalog()
        ply = self.root / "vggsfm/outputs/fixture/point_cloud.ply"
        write_binary_ply(ply, 2)
        manifest = self.root / "vggsfm/outputs/fixture/manifest.json"
        manifest.write_text(json.dumps({
            "status": "complete",
            "run_id": "fixture-vgg",
            "point_cloud": {"point_count": 2},
            "files": [{
                "path": "point_cloud.ply",
                "bytes": ply.stat().st_size,
                "sha256": hashlib.sha256(ply.read_bytes()).hexdigest(),
            }],
        }), encoding="utf-8")
        views = self.root / "vgg_views.json"
        views.write_text(json.dumps({"schema_version": 1, "views": [{
            "id": "VGGSFM", "title": "Fixture learned SfM",
            "stage_type": "learned_reconstruction", "method": "vggsfm",
            "subjects": {
                "light": {
                    "available": True, "point_count": 2,
                    "relative_ply": "vggsfm/outputs/fixture/point_cloud.ply",
                    "manifest_relative": "vggsfm/outputs/fixture/manifest.json",
                    "manifest_run_id": "fixture-vgg",
                },
                "dark": {
                    "available": False, "point_count": None, "relative_ply": None,
                    "reason": "fixture dark is not complete",
                },
            },
        }]}), encoding="utf-8")
        selected = DesktopSelection(catalog, views).resolve("VGGSFM", "light")
        self.assertTrue(selected["validation"]["valid"])
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["status"] = "running"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CatalogError, "not the configured complete"):
            DesktopSelection(catalog, views).resolve("VGGSFM", "light")

    def test_cli_list_and_path_are_machine_reusable(self) -> None:
        expected_path = (
            self.root
            / "sfm/historical/transferred/sfm/reconstructions/experiments/"
            "E10_quality_exhaustive_guided_consensus90/light_shirt/subject.ply"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--data-root", str(self.root), "list", "--json"])
        self.assertEqual(code, 0)
        rows = json.loads(output.getvalue())
        self.assertEqual(len(rows), 28)
        self.assertEqual(
            [(row["experiment"], row["subject"]) for row in rows[-8:]],
            [
                ("MVS", "light"), ("MVS", "dark"),
                ("MVSCLEAN", "light"), ("MVSCLEAN", "dark"),
                ("VGGSFM", "light"), ("VGGSFM", "dark"),
                ("VGGCLEAN", "light"), ("VGGCLEAN", "dark"),
            ],
        )

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--data-root", str(self.root), "path", "E10", "--subject", "light"])
        self.assertEqual(code, 0)
        self.assertEqual(Path(output.getvalue().strip()), expected_path)


class HistoricalHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        payload = catalog_payload()
        payload["reconstruction_root"] = "historical/reconstructions"
        payload["experiments"][0]["subjects"]["light"]["point_count"] = 2
        self.historical = HistoricalCatalog(
            self.root, write_catalog(self.root, payload)
        )
        self.views_path = write_desktop_views(self.root, 3)
        self.catalog = CombinedBrowserCatalog(self.historical, self.views_path)
        self.result = self.catalog.result("E1", "light")
        write_binary_ply(self.result.ply_path, 2)
        self.mvs_result = self.catalog.result("MVS", "light")
        write_binary_ply(self.mvs_result.ply_path, 3)
        self.server: HTTPServer = ReusableThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.catalog)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_catalog_static_ui_and_cloud_routes(self) -> None:
        with urlopen(f"{self.base}/api/catalog") as response:
            payload = json.load(response)
        self.assertEqual(len(payload["experiments"]), 11)
        self.assertEqual(payload["experiments"][-1]["id"], "MVS")
        self.assertEqual(
            payload["experiments"][-1]["subjects"]["light"]["point_count"], 3
        )
        self.assertFalse(
            payload["experiments"][-1]["subjects"]["dark"]["available"]
        )

        with urlopen(f"{self.base}/") as response:
            html = response.read().decode("utf-8")
        self.assertIn("SfM history", html)
        self.assertIn("viewer.js", html)
        self.assertIn('id="background-preset"', html)
        self.assertIn('value="custom"', html)
        self.assertIn('id="background-custom"', html)

        with urlopen(f"{self.base}/viewer.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("cv802.saved-viewer.background.v1", script)
        self.assertIn("gl.clearColor(red, green, blue, 1)", script)

        with urlopen(f"{self.base}/cloud/E1/light.ply") as response:
            self.assertEqual(response.read(3), b"ply")
        with urlopen(f"{self.base}/cloud/MVS/light.ply") as response:
            body = response.read()
        self.assertTrue(body.startswith(b"ply\n"))
        self.assertEqual(read_ply_header(self.mvs_result.ply_path)["vertex_count"], 3)

    def test_unavailable_and_unrecognized_routes_fail_closed(self) -> None:
        for path in (
            "/cloud/E10/dark.ply",
            "/cloud/MVS/dark.ply",
            "/cloud/MVS/../../outside.ply",
            "/cloud/E1/../../outside.ply",
            "/not-a-route",
        ):
            with self.subTest(path=path), self.assertRaises(HTTPError) as caught:
                urlopen(f"{self.base}{path}")
            self.assertEqual(caught.exception.code, 404)

    def test_mvs_config_is_reloaded_and_paths_remain_method_scoped(self) -> None:
        with urlopen(f"{self.base}/api/catalog") as response:
            first = json.load(response)
        self.assertFalse(first["experiments"][-1]["subjects"]["dark"]["available"])

        payload = json.loads(self.views_path.read_text(encoding="utf-8"))
        payload["views"][0]["subjects"]["dark"] = {
            "available": True,
            "point_count": 4,
            "relative_ply": "mvs/experiment/outputs/dark-fused.ply",
        }
        self.views_path.write_text(json.dumps(payload), encoding="utf-8")
        dark = self.root / "mvs/experiment/outputs/dark-fused.ply"
        write_binary_ply(dark, 4)
        with urlopen(f"{self.base}/api/catalog") as response:
            refreshed = json.load(response)
        self.assertTrue(
            refreshed["experiments"][-1]["subjects"]["dark"]["available"]
        )
        with urlopen(f"{self.base}/api/validate/MVS/dark") as response:
            validation = json.load(response)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["ply"]["vertex_count"], 4)

        payload["views"][0]["subjects"]["dark"]["relative_ply"] = (
            "sfm/historical/escape.ply"
        )
        self.views_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(HTTPError) as caught:
            urlopen(f"{self.base}/cloud/MVS/dark.ply")
        self.assertEqual(caught.exception.code, 404)

    def test_public_network_bind_is_rejected(self) -> None:
        with self.assertRaisesRegex(CatalogError, "loopback"):
            serve(self.catalog, "0.0.0.0", 8765)


if __name__ == "__main__":
    unittest.main()
