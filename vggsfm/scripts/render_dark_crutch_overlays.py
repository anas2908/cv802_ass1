#!/usr/bin/env python3
"""Render six audit overlays for dark VGGSfM crutch-corridor protection."""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggsfm_engine.io_utils import atomic_write_json, file_record, object_sha256, read_json
from vggsfm_engine.paths import production_layout
from vggsfm_engine.projection_cleanup import (
    corridor_hits,
    project_original_pixels,
    separated_view_confirmation,
)


SOURCE_RUN = "dark-shirt-vggsfm-all290-a10040-fit-v2"
CLEAN_RUN = SOURCE_RUN + "-mask-crutch-clean-v1"
INPUT_DATASET = "dark_shirt_cleanup_inputs"


def render() -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("overlay rendering must run inside the active allocation")
    layout = production_layout()
    layout.create_runtime_directories()
    os.environ.update(layout.runtime_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_root = layout.output_root(CLEAN_RUN)
    audit_path = layout.root / "evaluation" / f"{CLEAN_RUN}-validation-v1/audit_receipt.json"
    audit = read_json(audit_path)
    cleanup = read_json(output_root / "cleanup_receipt.json")
    if audit.get("status") != "complete" or cleanup.get("status") != "complete":
        raise RuntimeError("cleanup and independent audit must be complete")
    destination = output_root / "overlays"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")

    input_root = layout.inputs / INPUT_DATASET
    corridors_path = input_root / "crutches/corridors.json"
    corridors = read_json(corridors_path)["annotated_image_corridors"]
    request_path = layout.experiment_root(SOURCE_RUN) / "request.json"
    request = read_json(request_path)
    official_to_source = {
        row["official"]: row["source"].removeprefix("images/")
        for row in request["official_image_name_map"]
    }
    source_to_official = {source: official for official, source in official_to_source.items()}
    if len(source_to_official) != 290 or not set(corridors).issubset(source_to_official):
        raise RuntimeError("corridor/image-name mapping is incomplete")

    import pycolmap

    model = layout.output_root(SOURCE_RUN) / "colmap/sparse/0"
    reconstruction = pycolmap.Reconstruction(str(model))
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    point_ids = np.asarray(sorted(reconstruction.points3D), dtype=np.uint64)
    xyz = np.stack([
        np.asarray(reconstruction.points3D[int(point_id)].xyz, dtype=np.float64)
        for point_id in point_ids
    ])
    names = sorted(corridors)
    pixels_by_name = {}
    usable_by_name = {}
    hits_by_name = {}
    centers = []
    for name in names:
        image = images_by_name[source_to_official[name]]
        camera = reconstruction.cameras[image.camera_id]
        pixels, usable = project_original_pixels(image, camera, xyz)
        hits = corridor_hits(
            pixels, usable, width=int(camera.width), height=int(camera.height),
            groups=corridors[name],
        )
        pixels_by_name[name] = pixels
        usable_by_name[name] = usable
        hits_by_name[name] = hits
        centers.append(np.asarray(image.projection_center(), dtype=np.float64))
    confirmed = separated_view_confirmation(
        xyz, np.stack(centers), np.stack([hits_by_name[name] for name in names]),
        minimum_views=3, minimum_pairwise_angle_degrees=15.0,
    )
    if int(confirmed.sum()) != cleanup["selection"]["crutch_selected_count"]:
        raise RuntimeError("overlay crutch selection differs from cleanup receipt")

    staging = output_root / f".overlays.{os.getpid()}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        rendered = []
        for index, name in enumerate(names):
            source_path = layout.inputs / "dark_shirt/images" / name
            mask_path = input_root / "masks" / Path(name).with_suffix(".png")
            with Image.open(source_path) as opened:
                original = opened.convert("RGB")
            width, height = original.size
            scale = min(1.0, 1200.0 / max(width, height))
            display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            display = original.resize(display_size, Image.Resampling.LANCZOS)
            with Image.open(mask_path) as opened:
                mask = opened.convert("L").resize(display_size, Image.Resampling.BILINEAR)
            rgba = display.convert("RGBA")
            body_alpha = mask.point(lambda value: 38 if value >= 128 else 0)
            body_layer = Image.new("RGBA", display_size, (30, 144, 255, 0))
            body_layer.putalpha(body_alpha)
            rgba = Image.alpha_composite(rgba, body_layer)

            corridor_layer = Image.new("RGBA", display_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(corridor_layer)
            for group in corridors[name].values():
                band = max(2, round(
                    2.0 * float(group["half_width_fraction_of_image_width"])
                    * width * scale
                ))
                for polyline in group["polylines"]:
                    points = [
                        (float(x) * width * scale, float(y) * height * scale)
                        for x, y in polyline
                    ]
                    draw.line(points, fill=(255, 128, 0, 90), width=band, joint="curve")
                    draw.line(points, fill=(255, 230, 0, 255), width=max(1, band // 8), joint="curve")
            rgba = Image.alpha_composite(rgba, corridor_layer)

            point_layer = Image.new("RGBA", display_size, (0, 0, 0, 0))
            point_draw = ImageDraw.Draw(point_layer)
            current = confirmed & hits_by_name[name] & usable_by_name[name]
            shown = pixels_by_name[name][current] * scale
            radius = max(1, round(2 * scale + 1))
            for x, y in shown:
                point_draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=(50, 255, 80, 220),
                )
            rgba = Image.alpha_composite(rgba, point_layer)
            caption = (
                f"{name} | orange/yellow: reviewed corridor | blue: body mask | "
                f"green: globally 3-view/15deg-confirmed XYZ hitting this view ({len(shown)})"
            )
            caption_layer = Image.new("RGBA", display_size, (0, 0, 0, 0))
            caption_draw = ImageDraw.Draw(caption_layer)
            caption_draw.rectangle((0, 0, display_size[0], 34), fill=(0, 0, 0, 205))
            caption_draw.text((8, 9), caption, fill=(255, 255, 255, 255), font=ImageFont.load_default())
            rgba = Image.alpha_composite(rgba, caption_layer)
            output = staging / f"overlay-{index:02d}.png"
            rgba.convert("RGB").save(output, format="PNG", optimize=True)
            rendered.append(rgba.convert("RGB"))
            rows.append({
                "image_relative": name,
                "raw_corridor_hit_count": int(hits_by_name[name].sum()),
                "confirmed_crutch_points_visible_in_corridor": int(current.sum()),
                "source_image": file_record(source_path),
                "mask": file_record(mask_path),
                "overlay": file_record(output, relative_to=staging),
            })

        thumb_width, thumb_height = 600, 800
        sheet = Image.new("RGB", (thumb_width * 3, thumb_height * 2), (15, 18, 24))
        for index, rendered_image in enumerate(rendered):
            thumbnail = rendered_image.copy()
            thumbnail.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            x = (index % 3) * thumb_width + (thumb_width - thumbnail.width) // 2
            y = (index // 3) * thumb_height + (thumb_height - thumbnail.height) // 2
            sheet.paste(thumbnail, (x, y))
        contact = staging / "contact-sheet.png"
        sheet.save(contact, format="PNG", optimize=True)
        receipt = {
            "schema_version": 1,
            "status": "rendered_pending_visual_review",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": CLEAN_RUN,
            "crutch_selected_count": int(confirmed.sum()),
            "annotation_image_count": len(names),
            "rule": "corridor hit in >=3 views with every confirming ray pair >=15 degrees",
            "legend": {
                "blue": "body mask foreground >=128",
                "orange_yellow": "reviewed 2D crutch corridor width/centerline",
                "green": "confirmed raw dark-VGGSfM XYZ also hitting this view corridor",
            },
            "rows_sha256": object_sha256(rows),
            "rows": rows,
            "contact_sheet": file_record(contact, relative_to=staging),
            "sources": {
                "cleanup_receipt": file_record(output_root / "cleanup_receipt.json"),
                "independent_audit": file_record(audit_path),
                "corridor_evidence": file_record(corridors_path := input_root / "crutches/corridors.json"),
                "source_code": file_record(Path(__file__)),
            },
            "limitations": [
                "Green points are existing raw XYZ selected by image-space evidence, not new/reconstructed crutch surfaces.",
                "The overlays do not perform occlusion testing.",
                "Corridor width can retain nearby body, hand, or floor points.",
            ],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        }
        atomic_write_json(staging / "overlay_receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return read_json(destination / "overlay_receipt.json")


def main() -> int:
    result = render()
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "crutch_selected_count": result["crutch_selected_count"],
        "annotation_image_count": result["annotation_image_count"],
        "contact_sheet": result["contact_sheet"],
        "rows": [
            {
                "image_relative": row["image_relative"],
                "raw_corridor_hit_count": row["raw_corridor_hit_count"],
                "confirmed_crutch_points_visible_in_corridor": row[
                    "confirmed_crutch_points_visible_in_corridor"
                ],
                "overlay": row["overlay"],
            }
            for row in result["rows"]
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
