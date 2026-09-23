#!/usr/bin/env python3
"""Prepare and run the transferred E1-E10 macOS recipes in data storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASSETS = HERE / "mac_pipeline_assets"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALIASES = {"light_shirt": "light_shirt", "dark_shirt": "black_shirt_crutches"}


def report_progress(percent: int, stage: str) -> None:
    """Emit a machine-readable UI update without hiding ordinary CLI output."""
    print("CV802_PROGRESS " + json.dumps({"percent": percent, "stage": stage}), flush=True)


def expected_result(root: Path, subject: str, experiment: str) -> Path:
    """Return the display PLY produced by an E1-E10 recipe."""
    reconstructions = root / "reconstructions"
    if experiment == "E1":
        return reconstructions / subject / "sparse_full_scene.ply"
    if experiment == "E2":
        return reconstructions / subject / "subject_preview/subject.ply"
    if experiment == "E3":
        return reconstructions / f"{subject}_quality/subject_preview/subject.ply"
    if experiment in ("E4", "E5"):
        threshold = "90" if experiment == "E4" else "97"
        if subject == "black_shirt_crutches":
            return reconstructions / (
                f"experiments/black_person_crutches_cleanup/"
                f"body_mask_consensus_{threshold}_crutches/subject.ply"
            )
        return reconstructions / (
            f"experiments/light_person_mask_cleanup/mask_consensus_{threshold}/subject.ply"
        )
    if experiment == "E6":
        return reconstructions / (
            f"experiments/E6_guided_off/{subject}_quality/subject_preview/subject.ply"
        )
    if experiment == "E7":
        return reconstructions / f"experiments/E7_guided_off_consensus90/{subject}/subject.ply"
    if experiment == "E8":
        return reconstructions / (
            f"experiments/E8_vocab_guided/{subject}_quality/subject_preview/subject.ply"
        )
    if experiment == "E9":
        return reconstructions / f"experiments/E9_vocab_guided_consensus90/{subject}/subject.ply"
    if experiment == "E10":
        return reconstructions / (
            f"experiments/E10_quality_exhaustive_guided_consensus90/{subject}/subject.ply"
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def run(command: list[object], cwd: Path, *, allow_existing: bool = False) -> None:
    printable = [str(part) for part in command]
    print("$ " + " ".join(printable), flush=True)
    result = subprocess.run(printable, cwd=cwd, env=dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        CV802_ALLOW_NON_SLURM="1"))
    if result.returncode and not allow_existing:
        raise RuntimeError(f"Command failed with status {result.returncode}")


def data_root() -> Path:
    value = os.environ.get("CV802_DATA_ROOT")
    if not value:
        raise RuntimeError("Set CV802_DATA_ROOT before running a Mac recipe")
    result = Path(value).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def images_for(dataset: str) -> Path:
    result = (REPO / "datasets" / dataset / "images").resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"Missing dataset images: {result}")
    if sum(p.is_file() and p.suffix.lower() in EXTENSIONS for p in result.rglob("*")) < 2:
        raise ValueError(f"Dataset has fewer than two supported images: {result}")
    return result


def workspace() -> Path:
    root = data_root() / "sfm" / "mac-e1-e10-workspace"
    root.mkdir(parents=True, exist_ok=True)
    if not ASSETS.is_dir():
        raise FileNotFoundError("The bundled mac_pipeline_assets folder is missing")
    shutil.copytree(ASSETS / "work", root / "work", dirs_exist_ok=True)
    shutil.copytree(ASSETS / "reconstructions", root / "reconstructions", dirs_exist_ok=True)
    shutil.copytree(HERE / "assignment1", root / "assignment1", dirs_exist_ok=True)
    environment = root / ".venv"
    if not environment.exists():
        environment.symlink_to(Path(sys.prefix).resolve(), target_is_directory=True)
    return root


def image_records(images: Path) -> list[dict]:
    result = []
    for path in sorted(p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS):
        relative = path.relative_to(images).as_posix()
        result.append({"image": relative, "original": path.name,
                       "kind": "video" if relative.startswith("video_") else "photo"})
    return result


def stage_dataset(root: Path, dataset: str) -> tuple[str, Path]:
    from PIL import Image

    subject = ALIASES.get(dataset, dataset)
    images = images_for(dataset)
    target = root / "reconstructions" / subject
    target.mkdir(parents=True, exist_ok=True)
    link = target / "images"
    if link.is_symlink() and link.resolve() != images:
        raise RuntimeError(f"Existing image link points elsewhere: {link}")
    if not link.exists():
        link.symlink_to(images, target_is_directory=True)
    records = image_records(images)
    (target / "input_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    groups = {}
    for folder in sorted({Path(row["image"]).parts[0] for row in records}):
        sample = next((images / folder).glob("*"))
        with Image.open(sample) as photo:
            width, height = photo.size
        groups[folder] = {"source": "video" if folder.startswith("video_") else "photo",
                          "width": width, "height": height}
    (target / "camera_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    return subject, target


def ensure_boxes(root: Path, subject: str, baseline: Path) -> Path:
    box_root = root / "work/full-capture/subject_boxes"
    box_root.mkdir(parents=True, exist_ok=True)
    output = box_root / f"{subject}_boxes.json"
    if output.is_file():
        return output
    rows = [{"dataset": subject, "image_name": row["image"],
             "path": str((baseline / "images" / row["image"]).resolve())}
            for row in json.loads((baseline / "input_manifest.json").read_text())]
    input_path = box_root / f"{subject}_input.json"
    raw_path = box_root / f"{subject}_raw_boxes.json"
    input_path.write_text(json.dumps(rows, indent=2) + "\n")
    binary = box_root / "detect_people"
    if not binary.is_file():
        run(["swiftc", box_root / "detect_people.swift", "-o", binary], root)
    run([binary, input_path, raw_path], root)
    detected = json.loads(raw_path.read_text())
    for row in detected:
        row["detector"] = "macOS Vision VNDetectHumanRectanglesRequest"
        selected = row.get("selected")
        if not selected:
            continue
        x0, y0, x1, y1 = selected["bbox_xyxy"]
        bw, bh = x1 - x0, y1 - y0
        if subject == "black_shirt_crutches":
            px, top, bottom = max(bw * .30, row["width"] * .035), bh * .055, max(bh * .14, row["height"] * .025)
        else:
            px, top, bottom = bw * .10, bh * .055, bh * .055
        row["padded_bbox_xyxy"] = [max(0, x0-px), max(0, y0-top),
                                    min(row["width"], x1+px), min(row["height"], y1+bottom)]
    output.write_text(json.dumps({row["image_name"]: row for row in detected}, indent=2) + "\n")
    return output


def ensure_e1(root: Path, subject: str, baseline: Path) -> None:
    if not (baseline / "reconstruction_report.json").is_file():
        run([sys.executable, root / "work/full-capture/run_reconstruction.py", subject], root)
    ensure_boxes(root, subject, baseline)


def ensure_e2(root: Path, subject: str, baseline: Path) -> None:
    ensure_e1(root, subject, baseline)
    preview = baseline / "subject_preview"
    if not (preview / "analysis.json").is_file():
        run([sys.executable, root / "work/full-capture/make_subject_preview.py", baseline,
             "--boxes", ensure_boxes(root, subject, baseline), "--output", preview,
             "--volume-fraction", ".8" if subject == "light_shirt" else ".9"], root)


def ensure_e3(root: Path, subject: str, baseline: Path) -> Path:
    ensure_e2(root, subject, baseline)
    native = root / "work/quality-sfm/photos_native" / subject
    native.mkdir(parents=True, exist_ok=True)
    native_images = native / "images"
    if not native_images.exists():
        native_images.symlink_to((baseline / "images").resolve(), target_is_directory=True)
    shutil.copy2(baseline / "camera_groups.json", native / "camera_groups.json")
    shutil.copy2(ensure_boxes(root, subject, baseline), native / "boxes.json")
    quality = root / "reconstructions" / f"{subject}_quality"
    if not (quality / "sfm_refine.json").is_file():
        run([sys.executable, root / "work/quality-sfm/prepare_quality_datasets.py", subject], root)
    if not (quality / "reconstruction_report.json").is_file():
        run([sys.executable, root / "work/quality-sfm/run_quality.py", f"{subject}_quality"], root)
    preview = quality / "subject_preview"
    if not (preview / "analysis.json").is_file():
        run([sys.executable, root / "work/full-capture/make_subject_preview.py", quality,
             "--boxes", native / "boxes.json", "--output", preview,
             "--volume-fraction", ".8" if subject == "light_shirt" else ".9"], root)
    return quality


def mask_manifest_rows(source: Path, mask_root: Path, dark: bool) -> list[dict]:
    rows = []
    for image in sorted(p for p in (source / "images").rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS):
        name = image.relative_to(source / "images").as_posix()
        prefix = "body_" if dark else ""
        rows.append({"image_name": name, "source": str(image.resolve()),
                     "raw_mask": str(mask_root / f"{prefix}raw_masks" / Path(name).with_suffix(".png")),
                     "raw_metadata": str(mask_root / f"{prefix}raw_metadata" / Path(name).with_suffix(".json")),
                     "mask": str(mask_root / f"{prefix}masks" / Path(name).with_suffix(".png"))})
    return rows


def approve_json(path: Path, **values: object) -> None:
    record = json.loads(path.read_text())
    record.update(values)
    path.write_text(json.dumps(record, indent=2) + "\n")


def ensure_masks(root: Path, subject: str, quality: Path) -> Path:
    dark = subject == "black_shirt_crutches"
    name = "black_person_crutches_cleanup" if dark else "light_person_mask_cleanup"
    mask_root = root / "reconstructions/experiments" / name
    manifest = mask_root / ("body_mask_manifest.json" if dark else "mask_manifest.json")
    if manifest.is_file():
        return mask_root
    rows = mask_manifest_rows(quality, mask_root, dark)
    input_name = "body_input_manifest.json" if dark else "input_manifest.json"
    (mask_root / input_name).write_text(json.dumps(rows, indent=2) + "\n")
    (mask_root / "pilot_manifest.json").write_text(json.dumps(rows[:6], indent=2) + "\n")
    source_name = "generate_body_masks.swift" if dark else "generate_person_masks.swift"
    binary_name = "generate_body_masks" if dark else "generate_person_masks"
    binary = mask_root / "scripts" / binary_name
    run(["swiftc", mask_root / "scripts" / source_name, "-o", binary], root)
    run([binary, mask_root / input_name], root)
    prepare = "prepare_body_mask_outputs.py" if dark else "prepare_mask_outputs.py"
    run([sys.executable, mask_root / "scripts" / prepare], root)
    return mask_root


def ensure_e4_e5(root: Path, subject: str, baseline: Path) -> None:
    quality = ensure_e3(root, subject, baseline)
    masks = ensure_masks(root, subject, quality)
    if subject == "light_shirt":
        if not (masks / "mask_consensus_90/analysis.json").is_file():
            run([sys.executable, masks / "filter_person_masks.py",
                 "--source", quality / "subject_preview"], root)
        return
    protection = masks / "crutch_protection"
    approved = protection / "protection_approved.json"
    if not approved.is_file():
        run([sys.executable, protection / "annotate_additional_rois.py"], root)
        run([sys.executable, protection / "build_protection.py"], root)
        record = json.loads((protection / "protection.json").read_text())
        record.update(status="approved_experimental_crutch_protection", ready_for_filter=True,
                      review={"status": "approved_for_experiment",
                              "basis": "Original manually reviewed normalized image corridors, re-evaluated on this fresh model"})
        approved.write_text(json.dumps(record, indent=2) + "\n")
    review = protection / "fresh_candidate_review"
    if not (review / "review.json").is_file():
        run([sys.executable, masks / "filter_person_crutches.py", "--source", quality / "subject_preview",
             "--manifest", masks / "body_mask_manifest.json", "--protection", approved,
             "--candidate-review", review, "--qc-only"], root)
        approve_json(review / "review.json", ready_for_filter=True, status="approved_for_experiment")
    if not (masks / "body_mask_consensus_90_crutches/analysis.json").is_file():
        run([sys.executable, masks / "filter_person_crutches.py", "--source", quality / "subject_preview",
             "--manifest", masks / "body_mask_manifest.json", "--protection", approved,
             "--candidate-review", review], root)


def run_mask_cleanup(root: Path, experiment: str, subject: str) -> None:
    target_name = {"E7": "E7_guided_off_consensus90",
                   "E9": "E9_vocab_guided_consensus90"}[experiment]
    target = root / "reconstructions/experiments" / target_name / subject
    if (target / "analysis.json").is_file():
        return
    command = [sys.executable, root / "work/matching-ablation/run_cleanup.py",
               experiment, subject]
    if subject == "black_shirt_crutches":
        review = root / "work/matching-ablation" / experiment / "crutch_review/review.json"
        if not review.is_file():
            run([*command, "--qc-only"], root)
            approve_json(review, ready_for_filter=True, status="approved_for_experiment")
    run(command, root)


def execute(dataset: str, experiment: str) -> Path:
    report_progress(2, "Preparing the reconstruction workspace")
    root = workspace()
    report_progress(6, "Checking the selected images")
    subject, baseline = stage_dataset(root, dataset)
    if experiment == "E10" and subject != "light_shirt":
        raise ValueError("E10 is intentionally light-shirt only")
    if subject not in ("light_shirt", "black_shirt_crutches") and experiment not in ("E1", "E2"):
        raise ValueError("New datasets support E1/E2 automatically; E3-E10 need a subject recipe profile")
    if experiment == "E1":
        report_progress(12, "E1 · extracting, matching, and reconstructing")
        ensure_e1(root, subject, baseline)
    elif experiment == "E2":
        report_progress(12, "Preparing the E1 baseline")
        ensure_e1(root, subject, baseline)
        report_progress(72, "E2 · filtering the subject rectangle")
        ensure_e2(root, subject, baseline)
    elif experiment in ("E3",):
        report_progress(12, "Preparing E1 and E2 prerequisites")
        ensure_e2(root, subject, baseline)
        report_progress(45, "E3 · higher-detail feature matching and triangulation")
        ensure_e3(root, subject, baseline)
    elif experiment in ("E4", "E5"):
        report_progress(12, "Preparing the E3 refined reconstruction")
        ensure_e3(root, subject, baseline)
        report_progress(62, f"{experiment} · generating masks and filtering points")
        ensure_e4_e5(root, subject, baseline)
    elif experiment in ("E6", "E7"):
        report_progress(12, "Preparing the E3 features and cameras")
        ensure_e3(root, subject, baseline)
        report_progress(48, "E6 · matching without guided matching")
        if not (root / f"reconstructions/experiments/E6_guided_off/{subject}_quality/reconstruction_report.json").is_file():
            run([sys.executable, root / "work/matching-ablation/E6/prepare_and_run.py", subject], root)
        if experiment == "E7":
            report_progress(72, "Preparing the person masks")
            ensure_e4_e5(root, subject, baseline)
            report_progress(88, "E7 · applying the 90% mask cleanup")
            run_mask_cleanup(root, "E7", subject)
    elif experiment in ("E8", "E9"):
        report_progress(12, "Preparing the E3 features and cameras")
        ensure_e3(root, subject, baseline)
        report_progress(48, "E8 · vocabulary matching and triangulation")
        target = root / f"reconstructions/experiments/E8_vocab_guided/{subject}_quality"
        if not (target / "reconstruction_report.json").is_file():
            run([sys.executable, root / "work/matching-ablation/prepare_vocab_pairs.py", subject], root)
            run([sys.executable, root / "work/matching-ablation/prepare_e8.py", subject], root)
            run([sys.executable, root / "work/matching-ablation/run_e8.py", subject], root)
        if experiment == "E9":
            report_progress(72, "Preparing the person masks")
            ensure_e4_e5(root, subject, baseline)
            report_progress(88, "E9 · applying the 90% mask cleanup")
            run_mask_cleanup(root, "E9", subject)
    elif experiment == "E10":
        report_progress(12, "Preparing the E3 reconstruction and masks")
        ensure_e4_e5(root, subject, baseline)
        report_progress(52, "Preparing all exhaustive guided pairs")
        internal = root / "work/E10-exhaustive-guided/datasets/light_shirt_quality"
        if not (internal / "sfm_refine.json").is_file():
            run([sys.executable, root / "work/E10-exhaustive-guided/prepare_e10.py", subject], root)
        report_progress(62, "E10 · exhaustive guided matching and reconstruction")
        run([sys.executable, root / "work/E10-exhaustive-guided/run_e10.py", "--subject", subject], root)
    report_progress(97, "Validating the reconstructed point cloud")
    result = expected_result(root, subject, experiment).resolve()
    if not result.is_file():
        raise RuntimeError(f"The recipe completed without its expected point cloud: {result}")
    receipt = {"dataset": dataset, "subject_recipe": subject, "experiment": experiment,
               "workspace": str(root), "source_images": str(images_for(dataset)),
               "result_ply": str(result),
               "recipe_assets_sha256": hashlib.sha256(str(ASSETS).encode()).hexdigest()}
    receipt_path = root / "receipts" / f"{dataset}_{experiment}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    report_progress(100, "Reconstruction complete")
    return receipt_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: mac_recipe_runner.py DATASET E1|...|E10")
    print(execute(sys.argv[1], sys.argv[2]))
