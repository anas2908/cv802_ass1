"""Validated E1-E10 metadata and safe paths into the permanent data root."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("/l/users/anas.khan/cv_802_ass1")
CATALOG_PATH = Path(__file__).resolve().parents[1] / "configs" / "historical_e1_e10.json"
SUBJECTS = ("light", "dark")
STAGE_TYPES = ("reconstruction", "derived_cleanup")


class CatalogError(ValueError):
    """Raised when catalog input or a requested result is unsafe or invalid."""


def _require_relative_path(value: str, field: str) -> Path:
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise CatalogError(f"{field} must be a non-empty relative path without '..'")
    return candidate


def _inside(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_join(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise CatalogError("refusing a path outside the historical reconstruction root")
    lexical = root / relative
    resolved_root = root.resolve(strict=False)
    resolved_target = lexical.resolve(strict=False)
    if not _inside(resolved_target, resolved_root):
        raise CatalogError("resolved result path escapes the historical reconstruction root")
    return lexical


@dataclass(frozen=True)
class ResultRef:
    """One experiment/subject display result."""

    experiment_id: str
    subject: str
    available: bool
    point_count: int | None
    relative_ply: Path | None
    ply_path: Path | None
    reason: str | None
    validation_manifest_path: Path | None = None
    validation_manifest_run_id: str | None = None
    publication_path: Path | None = None
    expected_ply_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "subject": self.subject,
            "available": self.available,
            "point_count": self.point_count,
            "relative_ply": str(self.relative_ply) if self.relative_ply else None,
            "ply_path": str(self.ply_path) if self.ply_path else None,
            "on_disk": bool(self.ply_path and self.ply_path.is_file()),
            "reason": self.reason,
            "manifest_bound": (
                self.validation_manifest_path is not None
                or self.publication_path is not None
            ),
            "publication_bound": self.publication_path is not None,
        }


class HistoricalCatalog:
    """Load immutable source metadata and resolve results below ``DATA_ROOT``."""

    def __init__(
        self,
        data_root: Path | str | None = None,
        catalog_path: Path | str = CATALOG_PATH,
    ) -> None:
        root_value = data_root or os.environ.get("CV802_DATA_ROOT") or DEFAULT_DATA_ROOT
        self.data_root = Path(root_value)
        if not self.data_root.is_absolute():
            raise CatalogError("data root must be an absolute path")

        source = Path(catalog_path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._validate(payload)
        self.payload = payload
        self.catalog_path = source
        root_relative = _require_relative_path(
            payload["reconstruction_root"], "reconstruction_root"
        )
        self.reconstruction_root = _safe_join(self.data_root, root_relative)
        self._by_id = {entry["id"]: entry for entry in payload["experiments"]}

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != 1:
            raise CatalogError("unsupported catalog schema")
        _require_relative_path(payload.get("reconstruction_root", ""), "reconstruction_root")
        experiments = payload.get("experiments")
        if not isinstance(experiments, list) or len(experiments) != 10:
            raise CatalogError("catalog must contain exactly E1-E10")
        expected_ids = [f"E{index}" for index in range(1, 11)]
        if [item.get("id") for item in experiments] != expected_ids:
            raise CatalogError("catalog experiments must be ordered E1 through E10")

        for experiment in experiments:
            if experiment.get("stage_type") not in STAGE_TYPES:
                raise CatalogError(f"invalid stage type for {experiment.get('id')}")
            if experiment.get("exact_rerun_portable") is not False:
                raise CatalogError("historical entries must not claim exact portable reruns")
            subjects = experiment.get("subjects")
            if not isinstance(subjects, dict) or tuple(subjects) != SUBJECTS:
                raise CatalogError(f"{experiment['id']} must declare light then dark")
            for subject in SUBJECTS:
                result = subjects[subject]
                if result.get("available") is True:
                    count = result.get("point_count")
                    if not isinstance(count, int) or count <= 0:
                        raise CatalogError(f"invalid point count for {experiment['id']} {subject}")
                    _require_relative_path(
                        result.get("relative_ply", ""),
                        f"{experiment['id']} {subject} relative_ply",
                    )
                elif result.get("available") is False:
                    if result.get("point_count") is not None or result.get("relative_ply") is not None:
                        raise CatalogError(f"unavailable {experiment['id']} {subject} has a result")
                    if not result.get("reason"):
                        raise CatalogError(f"unavailable {experiment['id']} {subject} needs a reason")
                else:
                    raise CatalogError(f"invalid availability for {experiment['id']} {subject}")

        cleanup_ids = {
            item["id"] for item in experiments if item["stage_type"] == "derived_cleanup"
        }
        if cleanup_ids != {"E2", "E4", "E5", "E7", "E9"}:
            raise CatalogError("derived cleanup stages must be exactly E2/E4/E5/E7/E9")
        if experiments[-1]["subjects"]["dark"]["available"]:
            raise CatalogError("E10 dark must remain unavailable")

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        key = experiment_id.upper()
        try:
            return self._by_id[key]
        except KeyError as exc:
            raise CatalogError(f"unknown experiment {experiment_id!r}; choose E1-E10") from exc

    def result(self, experiment_id: str, subject: str, require_available: bool = True) -> ResultRef:
        subject_key = subject.lower()
        if subject_key not in SUBJECTS:
            raise CatalogError("subject must be 'light' or 'dark'")
        experiment = self.experiment(experiment_id)
        raw = experiment["subjects"][subject_key]
        if not raw["available"]:
            if require_available:
                raise CatalogError(raw["reason"])
            return ResultRef(experiment["id"], subject_key, False, None, None, None, raw["reason"])
        relative = _require_relative_path(raw["relative_ply"], "relative_ply")
        path = _safe_join(self.reconstruction_root, relative)
        return ResultRef(
            experiment["id"], subject_key, True, raw["point_count"], relative, path, None
        )

    def experiments(self) -> Iterable[dict[str, Any]]:
        return iter(self.payload["experiments"])

    def describe(self, experiment_id: str) -> dict[str, Any]:
        source = self.experiment(experiment_id)
        description = {key: value for key, value in source.items() if key != "subjects"}
        description["inspectable_preserved_result"] = True
        description["subjects"] = {
            subject: self.result(source["id"], subject, require_available=False).as_dict()
            for subject in SUBJECTS
        }
        return description

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.payload["schema_version"],
            "series_note": self.payload["series_note"],
            "data_root": str(self.data_root),
            "reconstruction_root": str(self.reconstruction_root),
            "experiments": [self.describe(item["id"]) for item in self.experiments()],
        }


def read_ply_header(path: Path, maximum_header_bytes: int = 65536) -> dict[str, Any]:
    """Read enough of a PLY to verify its declared vertex count and RGB fields."""

    with path.open("rb") as handle:
        header = bytearray()
        while len(header) <= maximum_header_bytes:
            line = handle.readline()
            if not line:
                raise CatalogError("PLY ended before end_header")
            header.extend(line)
            if line.strip() == b"end_header":
                break
        else:
            raise CatalogError("PLY header exceeds the safety limit")

    try:
        lines = header.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise CatalogError("PLY header is not ASCII") from exc
    if not lines or lines[0] != "ply":
        raise CatalogError("file does not start with the PLY magic")
    formats: list[str] = []
    for line in lines:
        if line.startswith("format "):
            fields = line.split(maxsplit=2)
            if len(fields) != 3:
                raise CatalogError("malformed PLY format declaration")
            formats.append(fields[1])
    if len(formats) != 1 or formats[0] not in {
        "ascii", "binary_little_endian", "binary_big_endian"
    }:
        raise CatalogError("unsupported or missing PLY format")

    vertex_count: int | None = None
    in_vertices = False
    vertex_properties: list[str] = []
    for line in lines:
        fields = line.split()
        if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
            try:
                vertex_count = int(fields[2])
            except ValueError as exc:
                raise CatalogError("invalid PLY vertex count") from exc
            in_vertices = True
        elif fields[:1] == ["element"]:
            in_vertices = False
        elif in_vertices and len(fields) >= 3 and fields[0] == "property":
            vertex_properties.append(fields[-1])
    if vertex_count is None or vertex_count < 0:
        raise CatalogError("PLY has no valid vertex element")
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = sorted(required.difference(vertex_properties))
    if missing:
        raise CatalogError(f"PLY vertex is missing properties: {', '.join(missing)}")
    return {
        "format": formats[0],
        "vertex_count": vertex_count,
        "vertex_properties": vertex_properties,
        "header_bytes": len(header),
    }


def validate_result(result: ResultRef) -> dict[str, Any]:
    if not result.available or result.ply_path is None:
        return {**result.as_dict(), "valid": False, "error": result.reason or "unavailable"}
    base = result.as_dict()
    try:
        if not result.ply_path.is_file():
            raise CatalogError("preserved PLY is missing")
        if result.validation_manifest_path is not None:
            try:
                manifest = json.loads(
                    result.validation_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise CatalogError(f"cannot validate saved VGGSfM result: {exc}") from exc
            if (
                manifest.get("status") != "complete"
                or manifest.get("run_id") != result.validation_manifest_run_id
                or manifest.get("point_cloud", {}).get("point_count") != result.point_count
            ):
                raise CatalogError("VGGSfM manifest is not the configured complete result")
            rows = []
            for row in manifest.get("files", []):
                declared = row.get("path")
                if not isinstance(declared, str):
                    continue
                declared_path = Path(declared)
                if (
                    declared == "point_cloud.ply"
                    or (declared_path.is_absolute() and declared_path.resolve(strict=False) == result.ply_path.resolve(strict=False))
                ):
                    rows.append(row)
            if len(rows) != 1 or result.validation_manifest_path.parent != result.ply_path.parent:
                raise CatalogError("VGGSfM manifest does not bind the configured PLY")
            digest = hashlib.sha256(result.ply_path.read_bytes()).hexdigest()
            if (
                result.ply_path.stat().st_size != rows[0].get("bytes")
                or digest != rows[0].get("sha256")
            ):
                raise CatalogError("VGGSfM PLY checksum differs from its complete manifest")
        if result.publication_path is not None:
            try:
                publication = json.loads(result.publication_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CatalogError(f"cannot validate saved publication: {exc}") from exc
            point_cloud = publication.get("point_cloud", {})
            if (
                publication.get("status") != "complete"
                or publication.get("point_count") != result.point_count
                or point_cloud.get("vertex_count") != result.point_count
                or point_cloud.get("sha256") != result.expected_ply_sha256
            ):
                raise CatalogError("saved publication is not the configured complete result")
            def artifact_path(value: object) -> Path:
                candidate = Path(str(value))
                if not candidate.is_absolute():
                    if not candidate.parts or ".." in candidate.parts:
                        raise CatalogError("publication contains an unsafe relative artifact path")
                    candidate = result.publication_path.parent / candidate
                return candidate.resolve(strict=False)

            published_path = artifact_path(point_cloud.get("path", ""))
            if (
                result.publication_path.parent != result.ply_path.parent
                or published_path != result.ply_path.resolve(strict=False)
            ):
                raise CatalogError("saved publication does not bind the configured PLY")
            artifact_rows = [
                row for row in publication.get("artifacts", [])
                if artifact_path(row.get("path", "")) == result.ply_path.resolve(strict=False)
            ]
            digest = hashlib.sha256(result.ply_path.read_bytes()).hexdigest()
            published_bytes = point_cloud.get("size_bytes", point_cloud.get("bytes"))
            if (
                len(artifact_rows) != 1
                or result.ply_path.stat().st_size != published_bytes
                or result.ply_path.stat().st_size != artifact_rows[0].get("bytes")
                or digest != result.expected_ply_sha256
                or digest != artifact_rows[0].get("sha256")
            ):
                raise CatalogError("saved PLY checksum differs from its publication")
        header = read_ply_header(result.ply_path)
        if header["vertex_count"] != result.point_count:
            raise CatalogError(
                f"vertex count {header['vertex_count']} does not match documented "
                f"count {result.point_count}"
            )
        return {**base, "valid": True, "ply": header, "size_bytes": result.ply_path.stat().st_size}
    except (CatalogError, OSError) as exc:
        return {**base, "valid": False, "error": str(exc)}
