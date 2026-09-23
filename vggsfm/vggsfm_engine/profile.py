"""Strict, allow-listed mapping from JSON profiles to Hydra overrides."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


@dataclass(frozen=True)
class InferenceProfile:
    """Reviewed knobs exposed by the CV802 wrapper.

    Visualization, ground-truth loading, output writing, checkpoint identity and
    scene location are deliberately not fields: the engine fixes them for
    headless, independent, reproducible inference.
    """

    name: str = "default"
    camera_type: str = "SIMPLE_RADIAL"
    shared_camera: bool = True
    query_frame_num: int = 6
    max_query_pts: int = 2048
    query_method: str = "aliked"
    fine_tracking: bool = True
    mixed_precision: str = "fp16"
    img_size: int = 1024
    robust_refine: int = 2
    bundle_adjustment_iterations: int = 1
    extra_point_pixel_interval: int = 10
    extra_point_neighbor_frames: int = 16
    concatenate_extra_points: bool = True
    seed: int = 0

    def validate(self, *, image_count: int | None = None) -> "InferenceProfile":
        integer_fields = {
            "query_frame_num": self.query_frame_num,
            "max_query_pts": self.max_query_pts,
            "img_size": self.img_size,
            "robust_refine": self.robust_refine,
            "bundle_adjustment_iterations": self.bundle_adjustment_iterations,
            "extra_point_pixel_interval": self.extra_point_pixel_interval,
            "extra_point_neighbor_frames": self.extra_point_neighbor_frames,
            "seed": self.seed,
        }
        if not isinstance(self.name, str) or not self.name or len(self.name) > 80:
            raise ConfigurationError("profile name must contain 1..80 characters")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields.values()
        ):
            raise ConfigurationError("profile integer fields must be JSON integers, not booleans")
        if not isinstance(self.camera_type, str):
            raise ConfigurationError("camera_type must be a string")
        if self.camera_type not in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
            raise ConfigurationError("camera_type must be SIMPLE_PINHOLE or SIMPLE_RADIAL")
        if not isinstance(self.shared_camera, bool):
            raise ConfigurationError("shared_camera must be a JSON boolean")
        if not isinstance(self.fine_tracking, bool):
            raise ConfigurationError("fine_tracking must be a JSON boolean")
        if not isinstance(self.concatenate_extra_points, bool):
            raise ConfigurationError("concatenate_extra_points must be a JSON boolean")
        if not 1 <= self.query_frame_num <= 64:
            raise ConfigurationError("query_frame_num must be in [1, 64]")
        if image_count is not None and self.query_frame_num > image_count:
            raise ConfigurationError(
                f"query_frame_num={self.query_frame_num} exceeds {image_count} input images"
            )
        if not 128 <= self.max_query_pts <= 16384:
            raise ConfigurationError("max_query_pts must be in [128, 16384]")
        if not isinstance(self.query_method, str):
            raise ConfigurationError("query_method must be a string")
        methods = self.query_method.split("+")
        if not methods or len(methods) != len(set(methods)) or any(
            item not in {"aliked", "sp", "sift"} for item in methods
        ):
            raise ConfigurationError(
                "query_method must be aliked, sp, sift, or a unique '+' combination"
            )
        if not isinstance(self.mixed_precision, str) or self.mixed_precision not in {
            "fp16",
            "bf16",
            "None",
        }:
            raise ConfigurationError("mixed_precision must be fp16, bf16, or None")
        if not 256 <= self.img_size <= 2048 or self.img_size % 8:
            raise ConfigurationError("img_size must be a multiple of 8 in [256, 2048]")
        if not 0 <= self.robust_refine <= 10:
            raise ConfigurationError("robust_refine must be in [0, 10]")
        if not 0 <= self.bundle_adjustment_iterations <= 10:
            raise ConfigurationError("bundle_adjustment_iterations must be in [0, 10]")
        interval = self.extra_point_pixel_interval
        if interval != -1 and not 2 <= interval <= 100:
            raise ConfigurationError("extra_point_pixel_interval must be -1 or in [2, 100]")
        neighbors = self.extra_point_neighbor_frames
        if neighbors != -1 and not 4 <= neighbors <= 400:
            raise ConfigurationError(
                "extra_point_neighbor_frames must be -1 or in [4, 400]"
            )
        if self.concatenate_extra_points and interval <= 0:
            raise ConfigurationError(
                "concatenate_extra_points requires a positive extra_point_pixel_interval"
            )
        if not 0 <= self.seed <= 2**31 - 1:
            raise ConfigurationError("seed must be in [0, 2^31-1]")
        return self

    @classmethod
    def from_json(cls, path: Path) -> "InferenceProfile":
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Cannot read profile {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigurationError("profile must be a JSON object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed - {"schema_version", "description"})
        if unknown:
            raise ConfigurationError(f"unknown profile fields: {', '.join(unknown)}")
        kwargs = {key: value for key, value in payload.items() if key in allowed}
        try:
            return cls(**kwargs).validate()
        except TypeError as exc:
            raise ConfigurationError(f"invalid profile fields: {exc}") from exc

    def with_overrides(self, **overrides: Any) -> "InferenceProfile":
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean).validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _hydra_bool(value: bool) -> str:
        return "True" if value else "False"

    def hydra_overrides(self) -> list[str]:
        """Translate reviewed fields to the official ``cfgs/demo.yaml`` names."""

        return [
            f"camera_type={self.camera_type}",
            f"shared_camera={self._hydra_bool(self.shared_camera)}",
            f"query_frame_num={self.query_frame_num}",
            f"max_query_pts={self.max_query_pts}",
            f"query_method={self.query_method}",
            f"fine_tracking={self._hydra_bool(self.fine_tracking)}",
            "mixed_precision='None'"
            if self.mixed_precision == "None"
            else f"mixed_precision={self.mixed_precision}",
            f"img_size={self.img_size}",
            f"robust_refine={self.robust_refine}",
            f"BA_iters={self.bundle_adjustment_iterations}",
            f"extra_pt_pixel_interval={self.extra_point_pixel_interval}",
            f"extra_by_neighbor={self.extra_point_neighbor_frames}",
            f"concat_extra_points={self._hydra_bool(self.concatenate_extra_points)}",
            f"seed={self.seed}",
        ]
