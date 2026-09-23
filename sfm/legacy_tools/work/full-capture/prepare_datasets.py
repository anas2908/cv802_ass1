"""Recreate the curated SfM input folders without overwriting existing files.

Uses the already converted still photos and selected video-frame manifest. It
does not decode the source archive or run reconstruction. Existing image bytes
and JSON content must match the expected inputs; their mtimes are preserved.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from collections import Counter
from pathlib import Path


WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
SUBJECTS = ("black_shirt_crutches", "light_shirt")


def load_json(path):
    return json.loads(path.read_text())


def dataset_plan(subject, photo_metadata, video_manifest):
    dataset = ROOT / "reconstructions" / subject
    photos = sorted(
        (item for item in photo_metadata
         if item["subject"] == subject and not item.get("exclude_recommended", False)),
        key=lambda item: (item["captured"], item.get("subsecond", "")),
    )
    frames = [item for item in video_manifest if item["subject"] == subject]
    manifest, groups, copies = [], {}, []
    for index, item in enumerate(photos):
        source = WORK / "photos" / "jpeg" / (Path(item["filename"]).stem + ".jpg")
        group = "photos_" + item["lens_group"].removeprefix(subject + "_")
        relative = f"{group}/{index:04d}_{source.name}"
        manifest.append({"image": relative, "original": item["filename"],
                         "kind": "photo", "captured": item["captured"]})
        groups[group] = {"source": "photo", "device": item["camera_model"],
                         "focal_length_35mm": item["focal_length_35mm"],
                         "width": item["width"], "height": item["height"]}
        copies.append((source, dataset / "images" / relative))
    for item in frames:
        clip = Path(item["source"]).stem
        source = WORK / "videos" / "frames" / subject / clip / Path(item["path"]).name
        group = "video_" + clip
        relative = f"{group}/{source.name}"
        manifest.append({"image": relative, "original": item["source"],
                         "kind": "video_frame", "time_seconds": item["time_seconds"],
                         "sharpness": item["sharpness"]})
        groups[group] = {"source": "video", "clip": item["source"],
                         "width": item["width"], "height": item["height"]}
        copies.append((source, dataset / "images" / relative))
    if len({item["image"] for item in manifest}) != len(manifest):
        raise ValueError(f"Duplicate output image paths for {subject}")
    return {"subject": subject, "dataset": dataset, "copies": copies,
            "manifest": manifest, "groups": groups}


def verify_existing(plan):
    """Preflight every source and existing destination before creating anything."""
    missing_images = []
    for source, destination in plan["copies"]:
        if not source.is_file():
            raise FileNotFoundError(f"Prepared source is missing: {source}")
        if destination.exists():
            if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
                raise ValueError(f"Existing image differs; left unchanged: {destination}")
        else:
            missing_images.append((source, destination))
    expected = {destination for _, destination in plan["copies"]}
    image_dir = plan["dataset"] / "images"
    actual = set(image_dir.rglob("*.jpg")) if image_dir.exists() else set()
    extras = actual - expected
    if extras:
        raise ValueError(f"Unexpected JPEG inputs; left unchanged: {sorted(extras)}")
    missing_json = []
    for filename, value in (("input_manifest.json", plan["manifest"]),
                            ("camera_groups.json", plan["groups"])):
        path = plan["dataset"] / filename
        if path.exists():
            if load_json(path) != value:
                raise ValueError(f"Existing JSON differs; left unchanged: {path}")
        else:
            missing_json.append((path, value))
    return missing_images, missing_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and report missing files without writing anything.")
    parser.add_argument("--subject", choices=("both",) + SUBJECTS, default="both")
    args = parser.parse_args()
    metadata = load_json(WORK / "photos" / "metadata.json")
    frames = load_json(WORK / "videos" / "frame_manifest.json")
    subjects = SUBJECTS if args.subject == "both" else (args.subject,)
    plans = [dataset_plan(subject, metadata, frames) for subject in subjects]
    checked = [(plan, *verify_existing(plan)) for plan in plans]
    for plan, missing_images, missing_json in checked:
        counts = Counter(item["kind"] for item in plan["manifest"])
        print(f"{plan['subject']}: {len(plan['manifest'])} inputs "
              f"({counts['photo']} photos + {counts['video_frame']} video frames), "
              f"{len(plan['groups'])} camera groups; "
              f"{len(missing_images)} missing images, {len(missing_json)} missing JSON files")
        if args.dry_run:
            continue
        for source, destination in missing_images:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects a file created after preflight.
            with source.open("rb") as source_file, destination.open("xb") as output:
                shutil.copyfileobj(source_file, output)
            shutil.copystat(source, destination)
        for path, value in missing_json:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as output:
                output.write(json.dumps(value, indent=2))
    print("Dry run complete; no files changed." if args.dry_run else
          "Preparation complete; existing matching files and their mtimes were preserved.")


if __name__ == "__main__":
    main()
