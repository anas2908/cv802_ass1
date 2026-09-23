"""Read-only browser catalog combining preserved SfM history with saved MVS views.

``HistoricalCatalog`` remains the strict E1--E10 authority. This adapter adds
only the small, separately configured saved-result entries used by the UI. A
new ``DesktopSelection`` is created for every public operation so an already
running browser observes an atomically updated config without restart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from historical_desktop import (
    DESKTOP_VIEWS_PATH,
    DesktopSelection,
    _safe_additional_result,
)

from .catalog import CatalogError, HistoricalCatalog, ResultRef, SUBJECTS


class CombinedBrowserCatalog:
    """Delegate E1--E10 and expose allow-listed validated saved results."""

    def __init__(
        self,
        historical: HistoricalCatalog,
        additional_views_path: Path = DESKTOP_VIEWS_PATH,
    ) -> None:
        self.historical = historical
        self.additional_views_path = additional_views_path
        self.data_root = historical.data_root
        self.reconstruction_root = historical.reconstruction_root

    def _selection(self) -> DesktopSelection:
        # Reload the small source config on every request. This never scans
        # result trees or imports Open3D; GUI imports stay lazy.
        return DesktopSelection(self.historical, self.additional_views_path)

    @staticmethod
    def _is_historical(experiment_id: str) -> bool:
        return experiment_id.upper() in {f"E{number}" for number in range(1, 11)}

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        if self._is_historical(experiment_id):
            return self.historical.experiment(experiment_id)
        selection = self._selection()
        try:
            return selection.experiment(experiment_id)
        except CatalogError as exc:
            raise CatalogError(
                f"unknown experiment {experiment_id!r}; choose E1-E10, MVS or VGGSFM"
            ) from exc

    def result(
        self,
        experiment_id: str,
        subject: str,
        require_available: bool = True,
    ) -> ResultRef:
        if self._is_historical(experiment_id):
            return self.historical.result(
                experiment_id, subject, require_available=require_available
            )
        subject_key = subject.lower()
        if subject_key not in SUBJECTS:
            raise CatalogError("subject must be 'light' or 'dark'")
        item = self.experiment(experiment_id)
        result = _safe_additional_result(self.data_root, item, subject_key)
        if require_available and not result.available:
            raise CatalogError(result.reason or "saved result is unavailable")
        return result

    def experiments(self) -> Iterable[dict[str, Any]]:
        # Materialize one stable view of this request. Historical entries are
        # immutable; only additional saved-result availability can refresh.
        selection = self._selection()
        return iter(
            [*self.historical.experiments()]
            + [
                selection.experiment(experiment_id)
                for experiment_id in selection.experiment_ids
                if not self._is_historical(experiment_id)
            ]
        )

    def describe(self, experiment_id: str) -> dict[str, Any]:
        if self._is_historical(experiment_id):
            return self.historical.describe(experiment_id)
        source = self.experiment(experiment_id)
        description = {key: value for key, value in source.items() if key != "subjects"}
        if source["method"] == "mvs" and source["stage_type"] == "derived_cleanup":
            metadata = {
                "method": "COLMAP MVS body-mask consensus with crutch protection",
                "camera_policy": (
                    "Uses the dark MVS experiment's calibrated cameras and raw dense XYZ"
                ),
                "display_variant": (
                    "Derived 90% body consensus union separately reviewed crutch protection"
                ),
            }
        elif source["method"] == "mvs":
            metadata = {
                "method": "COLMAP CUDA PatchMatch Stereo and stereo fusion",
                "camera_policy": "Uses a separately staged, validated sparse COLMAP model",
                "display_variant": "Validated coloured dense fused PLY",
            }
        elif source["stage_type"] == "derived_cleanup":
            metadata = {
                "method": "VGGSfM native-mask projection consensus (derived, CPU only)",
                "camera_policy": (
                    "Projects raw points with independently estimated VGGSfM cameras; no E10 poses"
                ),
                "display_variant": (
                    "90% body-mask consensus; the dark result additionally unions "
                    "independently verified crutch-corridor candidates"
                ),
            }
        else:
            metadata = {
                "method": "Official VGGSfM v2 (independent learned SfM)",
                "camera_policy": "Cameras and geometry were estimated independently from raw images",
                "display_variant": (
                    "Full raw coloured PLY: tracked SfM core plus official trackless grid extras"
                ),
            }
        description.update({
            **metadata,
            "exact_rerun_portable": False,
            "inspectable_preserved_result": True,
        })
        description["subjects"] = {
            subject: self.result(
                source["id"], subject, require_available=False
            ).as_dict()
            for subject in SUBJECTS
        }
        return description

    def public_payload(self) -> dict[str, Any]:
        historical = self.historical.public_payload()
        return {
            **historical,
            "series_note": (
                f"{historical['series_note']} MVS, its labelled cleanup, and VGGSfM are additional validated "
                "saved method results, not E11 historical SfM experiments."
            ),
            "experiments": [
                self.describe(item["id"]) for item in self.experiments()
            ],
        }
