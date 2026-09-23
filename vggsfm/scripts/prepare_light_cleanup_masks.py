#!/usr/bin/env python3
"""Copy and verify the 125 light-person masks into VGGSfM method storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.io_utils import atomic_write_json, file_record, object_sha256
from vggsfm_engine.paths import production_layout, require_within


SOURCE = Path(
    "/l/users/anas.khan/cv_802_ass1/sfm/historical/transferred/sfm/"
    "reconstructions/experiments/light_person_mask_cleanup"
)
DESTINATION_NAME = "light_shirt_cleanup_masks"
REQUEST = Path(
    "/l/users/anas.khan/cv_802_ass1/vggsfm/experiments/"
    "light-shirt-vggsfm-v1/request.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare() -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("mask preparation must run inside the active Slurm allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    destination = layout.inputs / DESTINATION_NAME
    if destination.exists():
        receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or receipt.get("mask_count") != 125:
            raise RuntimeError("existing method-local mask dataset has no complete receipt")
        return receipt

    source_root = SOURCE.resolve(strict=True)
    source_masks = (source_root / "masks").resolve(strict=True)
    source_manifest = (source_root / "mask_manifest.json").resolve(strict=True)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    mapping = request.get("official_image_name_map", [])
    source_names = {row["source"].removeprefix("images/") for row in mapping}
    rows_by_name = manifest.get("images", {})
    if (
        manifest.get("complete") is not True
        or manifest.get("ready_for_filter") is not True
        or manifest.get("summary", {}).get("usable") != 125
        or set(rows_by_name) != source_names
        or len(mapping) != 125
    ):
        raise RuntimeError("historical mask manifest does not exactly cover VGGSfM inputs")

    staging = layout.inputs / f".{DESTINATION_NAME}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    layout.assert_member(staging, label="mask staging")
    (staging / "masks").mkdir(parents=True)
    rows = []
    try:
        for name in sorted(source_names):
            source_row = rows_by_name[name]
            if source_row.get("status") != "valid" or source_row.get("usable") is not True:
                raise RuntimeError(f"mask is not valid/usable: {name}")
            relative_mask = Path(source_row["mask_relative_path"])
            if relative_mask.parts[0] != "masks" or ".." in relative_mask.parts:
                raise RuntimeError(f"unsafe historical mask path: {relative_mask}")
            source = require_within(
                source_root / relative_mask, source_masks,
                label="historical light mask", must_exist=True,
            )
            target_relative = Path(name).with_suffix(".png")
            target = staging / "masks" / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.copying")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
            digest = sha256(target)
            if digest != source_row["mask_sha256"] or digest != sha256(source):
                raise RuntimeError(f"mask checksum mismatch: {name}")
            with Image.open(target) as image:
                if image.mode != "L" or image.size != (
                    source_row["width"], source_row["height"]
                ) or image.getextrema() != (0, 255):
                    raise RuntimeError(f"unexpected mask mode/dimensions/range: {name}")
            rows.append({
                "image_relative": name,
                "mask_relative": (Path("masks") / target_relative).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": digest,
                "width": source_row["width"],
                "height": source_row["height"],
                "mode": "L",
                "minimum": 0,
                "maximum": 255,
            })
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": DESTINATION_NAME,
            "mask_count": len(rows),
            "image_count": len(source_names),
            "exact_input_name_coverage": True,
            "all_masks_mode_l_native_dimensions_range_0_255": True,
            "semantics": manifest["method"]["value_semantics"],
            "threshold_policy": "foreground iff mask uint8 value >= 128",
            "source_manifest": file_record(source_manifest),
            "vggsfm_request": file_record(REQUEST),
            "rows_sha256": object_sha256(rows),
            "rows": rows,
            "provenance_note": (
                "Copied from transferred historical Apple Vision masks solely as image-space "
                "segmentation evidence; cleanup uses independent VGGSfM cameras/geometry."
            ),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        }
        atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return receipt


def main() -> int:
    result = prepare()
    print(json.dumps({key: result[key] for key in (
        "status", "dataset", "mask_count", "image_count", "rows_sha256"
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
