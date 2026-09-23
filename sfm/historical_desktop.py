"""Read-only Open3D desktop viewer for the preserved SfM E1-E10 results.

This launcher reuses the assignment's original ``AppWindow`` and
``SceneWidget``.  It adds experiment/subject selectors but deliberately loads
only catalog-approved PLY files; it never calls feature extraction, matching,
mapping, triangulation, or a copied historical runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from historical_browser.catalog import (
    CatalogError,
    HistoricalCatalog,
    ResultRef,
    SUBJECTS,
    validate_result,
)


SUBJECT_LABELS = {
    "light": "Light shirt",
    "dark": "Black shirt + crutches",
}
STATUS_SUBJECT_LABELS = {"light": "Light", "dark": "Black"}
DESKTOP_VIEWS_PATH = Path(__file__).resolve().parent / "configs" / "desktop_saved_views.json"


def _safe_additional_result(data_root: Path, view: dict[str, Any], subject: str) -> ResultRef:
    raw = view["subjects"][subject]
    if not raw["available"]:
        return ResultRef(view["id"], subject, False, None, None, None, raw["reason"])

    relative = Path(raw["relative_ply"])
    if relative.is_absolute() or ".." in relative.parts:
        raise CatalogError(f"unsafe additional saved-result path for {view['id']} {subject}")
    method_root = (data_root / view["method"]).resolve(strict=False)
    target = (data_root / relative).resolve(strict=False)
    try:
        target.relative_to(method_root)
    except ValueError as exc:
        raise CatalogError(
            f"{view['id']} {subject} path escapes DATA_ROOT/{view['method']}"
        ) from exc
    validation_manifest = None
    validation_run_id = None
    publication_path = None
    expected_ply_sha256 = None
    if view["method"] == "vggsfm":
        manifest_relative = Path(raw["manifest_relative"])
        manifest_path = (data_root / manifest_relative).resolve(strict=False)
        try:
            manifest_path.relative_to(method_root)
        except ValueError as exc:
            raise CatalogError("VGGSfM validation manifest escapes DATA_ROOT/vggsfm") from exc
        validation_manifest = manifest_path
        validation_run_id = raw["manifest_run_id"]
    if "publication_relative" in raw or "ply_sha256" in raw:
        publication_relative = Path(raw["publication_relative"])
        publication_path = (data_root / publication_relative).resolve(strict=False)
        try:
            publication_path.relative_to(method_root)
        except ValueError as exc:
            raise CatalogError("saved-result publication escapes its method data root") from exc
        expected_ply_sha256 = raw["ply_sha256"]
    return ResultRef(
        view["id"],
        subject,
        True,
        raw["point_count"],
        relative,
        target,
        None,
        validation_manifest,
        validation_run_id,
        publication_path,
        expected_ply_sha256,
    )


def _load_additional_views(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read desktop saved-view config {path}: {exc}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("views"), list):
        raise CatalogError("unsupported desktop saved-view config")

    views: dict[str, dict[str, Any]] = {}
    for view in payload["views"]:
        view_id = view.get("id")
        if not isinstance(view_id, str) or not view_id or view_id in views:
            raise CatalogError("desktop saved-view IDs must be unique non-empty strings")
        if view_id in {f"E{index}" for index in range(1, 11)}:
            raise CatalogError("additional saved views cannot replace E1-E10")
        allowed_stages = {
            "mvs": {"dense_reconstruction", "derived_cleanup"},
            "vggsfm": {"learned_reconstruction", "derived_cleanup"},
        }.get(view.get("method"), set())
        if view.get("stage_type") not in allowed_stages:
            raise CatalogError(f"invalid stage type for additional view {view_id}")
        if not isinstance(view.get("title"), str) or not view["title"]:
            raise CatalogError(f"additional view {view_id} needs a title")
        subjects = view.get("subjects")
        if not isinstance(subjects, dict) or tuple(subjects) != SUBJECTS:
            raise CatalogError(f"additional view {view_id} must declare light then dark")
        for subject in SUBJECTS:
            result = subjects[subject]
            if result.get("available") is True:
                if not isinstance(result.get("point_count"), int) or result["point_count"] <= 0:
                    raise CatalogError(f"invalid point count for {view_id} {subject}")
                relative = result.get("relative_ply")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                ):
                    raise CatalogError(f"invalid path for {view_id} {subject}")
                if view["method"] == "vggsfm":
                    manifest_relative = result.get("manifest_relative")
                    manifest_run_id = result.get("manifest_run_id")
                    if (
                        not isinstance(manifest_relative, str)
                        or not manifest_relative
                        or Path(manifest_relative).is_absolute()
                        or ".." in Path(manifest_relative).parts
                        or not manifest_relative.startswith("vggsfm/")
                    ):
                        raise CatalogError(f"invalid VGGSfM manifest path for {view_id} {subject}")
                    if not isinstance(manifest_run_id, str) or not manifest_run_id:
                        raise CatalogError(f"missing VGGSfM manifest run ID for {view_id} {subject}")
                has_publication = "publication_relative" in result or "ply_sha256" in result
                if has_publication:
                    publication_relative = result.get("publication_relative")
                    digest = result.get("ply_sha256")
                    if (
                        not isinstance(publication_relative, str)
                        or not publication_relative
                        or Path(publication_relative).is_absolute()
                        or ".." in Path(publication_relative).parts
                        or not publication_relative.startswith(f"{view['method']}/")
                        or not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                    ):
                        raise CatalogError(
                            f"invalid publication binding for {view_id} {subject}"
                        )
            elif result.get("available") is False:
                if result.get("point_count") is not None or result.get("relative_ply") is not None:
                    raise CatalogError(f"unavailable {view_id} {subject} has result metadata")
                if not result.get("reason"):
                    raise CatalogError(f"unavailable {view_id} {subject} needs a reason")
            else:
                raise CatalogError(f"invalid availability for {view_id} {subject}")
        views[view_id] = view
    return views


class DesktopSelection:
    """UI-independent, testable selection logic for the desktop window."""

    def __init__(
        self,
        catalog: HistoricalCatalog,
        additional_views_path: Path = DESKTOP_VIEWS_PATH,
    ) -> None:
        self.catalog = catalog
        self.additional_views_path = additional_views_path
        self._additional = _load_additional_views(additional_views_path)
        self.experiment_ids = (
            *(item["id"] for item in catalog.experiments()),
            *self._additional,
        )

    def refresh(self) -> None:
        """Reload only the small source config; never scan or mutate data."""

        additional = _load_additional_views(self.additional_views_path)
        self._additional = additional
        self.experiment_ids = (
            *(item["id"] for item in self.catalog.experiments()),
            *additional,
        )

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        key = experiment_id.upper()
        if key in self._additional:
            return self._additional[key]
        return self.catalog.experiment(key)

    def experiment_label(self, experiment_id: str) -> str:
        item = self.experiment(experiment_id)
        if item["stage_type"] == "dense_reconstruction":
            kind = "dense MVS"
        elif item["stage_type"] == "learned_reconstruction":
            kind = "learned SfM"
        else:
            kind = "cleanup" if item["stage_type"] == "derived_cleanup" else "reconstruction"
        return f"{item['id']} — {item['title']} [{kind}]"

    def available_subjects(self, experiment_id: str) -> tuple[str, ...]:
        key = experiment_id.upper()
        if key in self._additional:
            return tuple(
                subject for subject in SUBJECTS if self._additional[key]["subjects"][subject]["available"]
            )
        return tuple(
            subject
            for subject in SUBJECTS
            if self.catalog.result(experiment_id, subject, require_available=False).available
        )

    def resolve(self, experiment_id: str, subject: str) -> dict[str, Any]:
        subject_key = subject.lower()
        if subject_key not in SUBJECTS:
            raise CatalogError("subject must be 'light' or 'dark'")
        item = self.experiment(experiment_id)
        if item["id"] in self._additional:
            result = _safe_additional_result(self.catalog.data_root, item, subject_key)
            if not result.available:
                raise CatalogError(result.reason or "saved result is unavailable")
        else:
            result = self.catalog.result(experiment_id, subject_key)
        report = validate_result(result)
        if not report["valid"]:
            raise CatalogError(report.get("error", "preserved result validation failed"))
        return {
            "experiment": item,
            "result": result,
            "validation": report,
        }


def _desktop_runtime() -> tuple[Any, Any, Any, type]:
    """Import GUI-only dependencies after non-GUI checks have succeeded."""

    assignment_root = Path(__file__).resolve().parent / "assignment1"
    if str(assignment_root) not in sys.path:
        sys.path.insert(0, str(assignment_root))

    try:
        import numpy as np
        import open3d as o3d
        import open3d.visualization.gui as gui
        from modules.gui.gui import AppWindow
    except ImportError as exc:
        raise RuntimeError(
            "The desktop viewer requires NumPy and Open3D with GUI support. "
            "Use the dedicated desktop environment and an active X/VNC display."
        ) from exc
    return np, o3d, gui, AppWindow


def build_window_class() -> tuple[Any, type]:
    """Return Open3D's GUI module and the read-only ``AppWindow`` subclass."""

    np, o3d, gui, AppWindow = _desktop_runtime()

    class HistoricalDesktopWindow(AppWindow):
        """Original starter window plus safe E1-E10 saved-result controls."""

        MODEL_NAME = "__historical_saved_ply__"
        PANEL_WIDTH_EM = 25

        def __init__(
            self,
            width: int,
            height: int,
            catalog: HistoricalCatalog,
            initial_experiment: str,
            initial_subject: str,
        ) -> None:
            super().__init__(width, height)
            self._historical = DesktopSelection(catalog)
            self._selected_experiment = initial_experiment
            self._selected_subject = initial_subject
            self._loaded_bounds = None
            self._loaded_cloud = None
            self._building_controls = True

            # Reconstruction controls remain part of the unchanged starter
            # source, but this dedicated window never exposes them.
            self.colmap_ctrls.visible = False
            self._camera_color.enabled = False
            self._camera_size.enabled = False

            em = self.window.theme.font_size
            controls = gui.CollapsableVert(
                "Saved SfM E1-E10 + MVS + VGGSfM (read only)",
                0.25 * em,
                gui.Margins(em, 0, 0, 0),
            )
            controls.add_child(gui.Label("Experiment"))
            self._historical_experiment = gui.Combobox()
            self._historical_experiment.tooltip = (
                "Choose one of the preserved E1-E10 experiment results. "
                "Cleanup and reconstruction stages are labelled explicitly."
            )
            self._experiment_labels: list[str] = []
            for experiment_id in self._historical.experiment_ids:
                label = self._historical.experiment_label(experiment_id)
                self._experiment_labels.append(label)
                self._historical_experiment.add_item(label)
            self._historical_experiment.set_on_selection_changed(
                self._on_historical_experiment
            )
            controls.add_child(self._historical_experiment)

            controls.add_child(gui.Label("Subject"))
            self._historical_subject = gui.Combobox()
            self._historical_subject.tooltip = (
                "Choose Light shirt or Black shirt + crutches when available."
            )
            self._historical_subject.set_on_selection_changed(
                self._on_historical_subject
            )
            controls.add_child(self._historical_subject)

            buttons = gui.Horiz(0.25 * em)
            self._historical_load = gui.Button("Load saved result")
            self._historical_load.tooltip = (
                "Validate and display the selected preserved PLY; no SfM is run."
            )
            self._historical_load.set_on_clicked(self.load_selected_result)
            self._historical_reset = gui.Button("Reset view")
            self._historical_reset.tooltip = "Fit the loaded cloud in the native Open3D camera."
            self._historical_reset.set_on_clicked(self.reset_historical_view)
            self._historical_reset.enabled = False
            buttons.add_child(self._historical_load)
            buttons.add_child(self._historical_reset)
            controls.add_child(buttons)

            self._historical_refresh = gui.Button("Refresh saved-result availability")
            self._historical_refresh.tooltip = (
                "Reload the small saved-view config after another validated result is published."
            )
            self._historical_refresh.set_on_clicked(self._refresh_saved_view_config)
            controls.add_child(self._historical_refresh)

            self._historical_status = gui.Label("Select a saved result, then load it.")
            self._historical_status.tooltip = "Current saved-result loading status."
            controls.add_child(self._historical_status)
            camera_note = gui.Label("Camera frustums are unavailable in the display PLY.")
            camera_note.tooltip = (
                "No camera geometry is invented; only the preserved colored points are shown."
            )
            controls.add_child(camera_note)
            controls.add_child(gui.Label("None are invented."))
            native_note = gui.Label("Native Open3D orbit, pan and wheel zoom are active.")
            native_note.tooltip = (
                "The SceneWidget has no custom browser-style zoom clamp."
            )
            controls.add_child(native_note)

            self._settings_panel.add_fixed(int(round(0.5 * em)))
            self._settings_panel.add_child(controls)
            self._historical_experiment.selected_text = self._historical.experiment_label(
                initial_experiment
            )
            self._refresh_subjects(initial_subject)
            self._building_controls = False
            self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            self.window.set_needs_layout()

        def _on_layout(self, layout_context: Any) -> None:
            """Keep the starter layout but widen this viewer's metadata panel."""

            content = self.window.content_rect
            self._scene.frame = content
            width = int(round(self.PANEL_WIDTH_EM * layout_context.theme.font_size))
            width = min(width, max(1, content.width - int(12 * layout_context.theme.font_size)))
            height = min(
                content.height,
                self._settings_panel.calc_preferred_size(
                    layout_context, gui.Widget.Constraints()
                ).height,
            )
            self._settings_panel.frame = gui.Rect(
                content.get_right() - width,
                content.y,
                width,
                height,
            )

        def _read_only_notice(self) -> None:
            self.window.show_message_box(
                "Read-only historical viewer",
                "Use the E1-E10 and subject selectors. Reconstruction, arbitrary "
                "folder loading, and export are disabled in this launcher.",
            )

        def _refresh_saved_view_config(self) -> None:
            try:
                current_experiment = self._selected_experiment
                current_subject = self._selected_subject
                self._historical.refresh()
                self._building_controls = True
                self._historical_experiment.clear_items()
                self._experiment_labels = []
                for experiment_id in self._historical.experiment_ids:
                    label = self._historical.experiment_label(experiment_id)
                    self._experiment_labels.append(label)
                    self._historical_experiment.add_item(label)
                if current_experiment not in self._historical.experiment_ids:
                    current_experiment = self._historical.experiment_ids[0]
                self._selected_experiment = current_experiment
                self._historical_experiment.selected_text = (
                    self._historical.experiment_label(current_experiment)
                )
                available = self._historical.available_subjects(current_experiment)
                if not available:
                    raise CatalogError(f"{current_experiment} has no available saved result")
                self._refresh_subjects(
                    current_subject if current_subject in available else available[0]
                )
                self._historical_status.text = "Availability refreshed; load the selected result."
                self._building_controls = False
                self.window.set_needs_layout()
                self.window.post_redraw()
            except (CatalogError, OSError, ValueError) as exc:
                self._building_controls = False
                self.window.show_message_box("Could not refresh saved results", str(exc))

        # These overrides are the callbacks registered by AppWindow.__init__.
        # They make the dedicated launcher fail closed even though the original
        # application and its Mac behavior remain untouched.
        def _on_menu_open_existing(self) -> None:
            self._read_only_notice()

        def _on_menu_open_image_folder(self) -> None:
            self._read_only_notice()

        def _on_menu_export(self) -> None:
            self._read_only_notice()

        def _on_fit_colmap_button(self) -> None:
            self._read_only_notice()

        def load_existing_result(self, data_path: str) -> None:
            del data_path
            self._read_only_notice()

        def _on_historical_experiment(self, name: str, index: int) -> None:
            del name
            if self._building_controls:
                return
            self._selected_experiment = self._historical.experiment_ids[index]
            available = self._historical.available_subjects(self._selected_experiment)
            preferred = self._selected_subject if self._selected_subject in available else available[0]
            self._refresh_subjects(preferred)
            self._selection_changed()

        def _refresh_subjects(self, preferred: str) -> None:
            available = self._historical.available_subjects(self._selected_experiment)
            self._historical_subject.clear_items()
            for subject in available:
                self._historical_subject.add_item(SUBJECT_LABELS[subject])
            self._selected_subject = preferred if preferred in available else available[0]
            self._historical_subject.selected_text = SUBJECT_LABELS[self._selected_subject]
            if self._selected_experiment == "E10":
                self._historical_status.text = "E10: Light only; Black was not run."
            elif self._selected_experiment == "MVS" and "dark" not in available:
                self._historical_status.text = "MVS: Light ready; Black is not complete yet."

        def _on_historical_subject(self, name: str, index: int) -> None:
            del name
            if self._building_controls:
                return
            available = self._historical.available_subjects(self._selected_experiment)
            self._selected_subject = available[index]
            self._selection_changed()

        def _selection_changed(self) -> None:
            self._scene.scene.clear_geometry()
            self._loaded_bounds = None
            self._loaded_cloud = None
            self._historical_reset.enabled = False
            if self._selected_experiment == "E10":
                self._historical_status.text = "E10: Black was not run; load Light."
            elif self._selected_experiment == "MVS" and "dark" not in self._historical.available_subjects("MVS"):
                self._historical_status.text = "MVS: Light ready; Black is not complete yet."
            else:
                self._historical_status.text = "Selection changed; load the saved result."
            self.window.post_redraw()

        def load_selected_result(self) -> None:
            try:
                selected = self._historical.resolve(
                    self._selected_experiment, self._selected_subject
                )
                result = selected["result"]
                cloud = o3d.io.read_point_cloud(str(result.ply_path))
                points = np.asarray(cloud.points)
                colors = np.asarray(cloud.colors)
                if points.shape != (result.point_count, 3):
                    raise CatalogError(
                        f"Open3D loaded {len(points)} points; expected {result.point_count}"
                    )
                if colors.shape != points.shape:
                    raise CatalogError("display PLY does not contain one RGB color per point")
                if not np.all(np.isfinite(points)) or not np.all(np.isfinite(colors)):
                    raise CatalogError("display PLY contains non-finite coordinates or colors")
                if np.any(colors < 0.0) or np.any(colors > 1.0):
                    raise CatalogError("Open3D decoded an RGB value outside [0, 1]")

                # Same display-only COLMAP-to-upright convention as the tested
                # WebGL viewer. This changes the in-memory view, never the PLY.
                display_transform = np.eye(4, dtype=np.float64)
                display_transform[1, 1] = -1.0
                display_transform[2, 2] = -1.0
                cloud.transform(display_transform)

                self._scene.scene.clear_geometry()
                self._scene.scene.add_geometry(self.MODEL_NAME, cloud, self.settings.material)
                self._loaded_cloud = cloud
                # Use the geometry's synchronous bounds.  The SceneWidget's
                # aggregate bounds can lag by one GUI frame under software GL.
                self._loaded_bounds = cloud.get_axis_aligned_bounding_box()
                self.reset_historical_view()
                self._historical_reset.enabled = True
                self._historical_status.text = (
                    f"Loaded {result.experiment_id} {STATUS_SUBJECT_LABELS[result.subject]}: "
                    f"{result.point_count:,} colored points."
                )
                self.window.post_redraw()
            except (CatalogError, OSError, RuntimeError, ValueError) as exc:
                self._historical_status.text = "Load failed."
                self.window.show_message_box("Saved result could not be loaded", str(exc))

        def reset_historical_view(self) -> None:
            if self._loaded_bounds is None:
                return
            self._scene.setup_camera(
                60.0,
                self._loaded_bounds,
                self._loaded_bounds.get_center(),
            )
            self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            self.window.post_redraw()

    return gui, HistoricalDesktopWindow


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data-root", type=Path)
    command.add_argument(
        "--experiment",
        choices=[*(f"E{i}" for i in range(1, 11)), "MVS"],
        default="E10",
    )
    command.add_argument("--subject", choices=SUBJECTS, default="light")
    command.add_argument("--width", type=int, default=1280)
    command.add_argument("--height", type=int, default=800)
    command.add_argument("--no-auto-load", action="store_true")
    command.add_argument(
        "--check-only",
        action="store_true",
        help="validate the chosen preserved PLY without importing Open3D or opening a window",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        catalog = HistoricalCatalog(data_root=arguments.data_root)
        selection = DesktopSelection(catalog)
        selected = selection.resolve(arguments.experiment, arguments.subject)
        if arguments.check_only:
            print(json.dumps(selected["validation"], indent=2, default=str))
            return 0
        if arguments.width < 640 or arguments.height < 480:
            raise CatalogError("desktop window must be at least 640 x 480")

        gui, window_type = build_window_class()
        gui.Application.instance.initialize()
        window = window_type(
            arguments.width,
            arguments.height,
            catalog,
            arguments.experiment,
            arguments.subject,
        )
        if not arguments.no_auto_load:
            window.load_selected_result()
        gui.Application.instance.run()
        return 0
    except (CatalogError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
