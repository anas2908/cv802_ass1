"""Fail-closed path policy for large MVS data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .constants import DATA_ROOT, METHOD_ROOT
from .errors import StoragePolicyError


_EXPERIMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_experiment_name(value: str) -> str:
    """Return a filesystem-safe experiment slug or reject it."""
    if not _EXPERIMENT_RE.fullmatch(value) or value in {".", ".."}:
        raise StoragePolicyError(
            "Experiment names may contain only letters, numbers, '.', '_' and '-' "
            "and must not contain a path separator"
        )
    return value


def safe_relative_path(value: str, label: str) -> Path:
    """Validate a portable path stored relative to an experiment directory."""
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise StoragePolicyError(f"{label} must be a clean relative path, got {value!r}")
    return path


@dataclass(frozen=True)
class PathPolicy:
    """Resolve and enforce the method's permanent data boundary.

    Production callers must use :meth:`production`.  ``for_tests`` exists so
    unit tests can exercise the same containment logic without writing to the
    cluster filesystem; the command-line interface never exposes it.
    """

    data_root: Path
    method_root: Path

    @classmethod
    def production(cls) -> "PathPolicy":
        return cls(DATA_ROOT.resolve(), METHOD_ROOT.resolve())

    @classmethod
    def for_tests(cls, root: Path) -> "PathPolicy":
        root = root.resolve()
        return cls(root, (root / "mvs").resolve())

    def assert_canonical(self) -> None:
        if self.data_root != DATA_ROOT.resolve() or self.method_root != METHOD_ROOT.resolve():
            raise StoragePolicyError("The production CLI must use the canonical MVS data root")

    def experiment(self, name: str) -> Path:
        path = (self.method_root / "experiments" / safe_experiment_name(name)).resolve()
        return self.require_method(path, "experiment")

    def require_method(
        self, path: Path | str, label: str, *, must_exist: bool | None = None
    ) -> Path:
        resolved = Path(path).expanduser().resolve(strict=False)
        if not _inside(resolved, self.method_root):
            raise StoragePolicyError(
                f"{label} must stay under {self.method_root}; resolved path was {resolved}"
            )
        if must_exist is True and not resolved.exists():
            raise StoragePolicyError(f"{label} does not exist: {resolved}")
        if must_exist is False and resolved.exists():
            raise StoragePolicyError(f"{label} already exists: {resolved}")
        return resolved

    def require_data_source(self, path: Path | str, label: str) -> Path:
        """Allow read-only staging sources anywhere under the assignment data root."""
        resolved = Path(path).expanduser().resolve(strict=True)
        if not _inside(resolved, self.data_root):
            raise StoragePolicyError(
                f"{label} must first be relocated under {self.data_root}; got {resolved}"
            )
        return resolved

    def from_experiment_relative(self, experiment: Path, value: str, label: str) -> Path:
        relative = safe_relative_path(value, label)
        resolved = (experiment / relative).resolve(strict=False)
        if not _inside(resolved, experiment):
            raise StoragePolicyError(f"{label} escapes experiment directory: {value!r}")
        return self.require_method(resolved, label)

