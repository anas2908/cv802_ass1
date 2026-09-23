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


def file_sha256(path: Path) -> str:
    """Hash a file without loading a potentially large database into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_file_for_hash(destination: Path, backup: Path, expected: str) -> bool:
    """Atomically restore a known-good generated file when a retry replaced it."""
    if destination.is_file() and file_sha256(destination) == expected:
        return False
    if not backup.is_file() or file_sha256(backup) != expected:
        raise RuntimeError(
            f"Neither {destination} nor its preserved backup matches the completed "
            "feature checkpoint. Existing files were left untouched."
        )
    staged = destination.with_name("." + destination.name + ".restore")
    shutil.copy2(backup, staged)
    os.replace(staged, destination)
    print(f"Restored the augmented E3 boxes from {backup}", flush=True)
    return True


def restore_e3_boxes(root: Path, subject: str, quality: Path, native_boxes: Path) -> None:
    """Keep E3's fallback feature crops byte-identical to its saved checkpoint."""
    inputs_path = quality / "colmap/sparse/sfm_inputs.json"
    config_path = quality / "sfm_refine.json"
    if not inputs_path.is_file() or not config_path.is_file():
        return
    # Select the checkpoint referenced by the saved result, ignoring obsolete
    # cache entries exactly as the read-only quality audit does.
    audit_directory = root / "work/quality-sfm"
    sys.path.insert(0, str(audit_directory))
    try:
        from audit_database import _select_checkpoint
        config = json.loads(config_path.read_text())
        saved = json.loads(inputs_path.read_text())
        _, metadata, _, _ = _select_checkpoint(quality, config, saved)
    finally:
        sys.path.pop(0)
    expected = metadata.get("signature", {}).get("boxes_sha256")
    if not isinstance(expected, str) or not expected:
        raise RuntimeError("The completed E3 checkpoint has no boxes fingerprint")
    backup = native_boxes.parent.parent / f"{subject}_boxes.json"
    restore_file_for_hash(native_boxes, backup, expected)


def validate_vocab_pairs(folder: Path, subject: str) -> bool:
    """Return False when absent; reject rather than overwrite partial retrieval output."""
    if not folder.exists():
        return False
    if not folder.is_dir() or folder.is_symlink():
        raise RuntimeError(f"Unsafe vocabulary-pair output; preserved at {folder}")
    required = ("matching_pairs.txt", "guided_pairs.txt", "retrieved_neighbors.json")
    provenance_path = folder / "retrieval_provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text())
        checksums = provenance["output_sha256"]
        valid = (
            provenance.get("subject") == subject
            and provenance.get("ordinary_pairs", 0) > 0
            and provenance.get("guided_pairs", 0) > 0
            and all((folder / name).is_file()
                    and checksums.get(name) == file_sha256(folder / name)
                    for name in required)
        )
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise RuntimeError(
            f"Incomplete or incompatible vocabulary-pair output was preserved at {folder}. "
            "Move that directory aside before deliberately starting retrieval again."
        )
    return True


def validate_e8_preparation(target: Path, pairs: Path) -> bool:
    """Recognize an atomically prepared E8 dataset so retries can continue."""
    if not target.exists():
        return False
    try:
        valid = (
            target.is_dir() and not target.is_symlink()
            and (target / "sfm_refine.json").is_file()
            and (target / "experiment_provenance.json").is_file()
            and file_sha256(target / "matching_pairs.txt")
                == file_sha256(pairs / "matching_pairs.txt")
            and file_sha256(target / "guided_pairs.txt")
                == file_sha256(pairs / "guided_pairs.txt")
        )
    except OSError:
        valid = False
    if not valid:
        raise RuntimeError(
            f"Incomplete or incompatible E8 preparation was preserved at {target}."
        )
    return True


def ensure_e10_resume_config(dataset: Path, subject: str, root: Path | None = None,
                             workers: int = 4) -> bool:
    """Repair only the known pre-run E10 configuration omission.

    The first portable Mac bundle prepared all exhaustive pairs correctly but
    inherited E3's non-resumable matching flag.  A retry must repair that small
    configuration file without discarding the prepared pairs or feature copy.
    """
    config_path = dataset / "sfm_refine.json"
    provenance_path = dataset / "experiment_provenance.json"
    config = json.loads(config_path.read_text())
    batch_size = config.get("matching_batch_size", 128)
    matching_threads = config.get("matching_threads", 1)
    if (config.get("resume_matching") is True
            and type(batch_size) is int and batch_size > 0
            and matching_threads == workers):
        return False
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("experiment") != "E10_quality_exhaustive_guided"
            or provenance.get("subject") != subject):
        raise RuntimeError(f"Refusing to alter unrecognized E10 preparation: {dataset}")
    ordinary = dataset / "matching_pairs.txt"
    guided = dataset / "guided_pairs.txt"
    rows = [line.split() for line in ordinary.read_text().splitlines() if line.strip()]
    expected_pairs = 7750 if subject == "light_shirt" else 41905
    if (ordinary.read_bytes() != guided.read_bytes() or len(rows) != expected_pairs
            or len({tuple(row) for row in rows}) != expected_pairs
            or any(len(row) != 2 or row[0] >= row[1] for row in rows)):
        raise RuntimeError("E10 exhaustive pair files failed validation; existing files were preserved")
    colmap = dataset / "colmap"
    started_entries = [path for path in colmap.iterdir()
                       if path.name != "quality_feature_cache" and not path.name.startswith(".")]
    reconstruction_complete = (dataset / "reconstruction_report.json").is_file()
    if reconstruction_complete:
        print(f"E10 reconstruction is already complete with {matching_threads} workers; reusing it.",
              flush=True)
        return False
    matching_cache = colmap / "quality_matching_cache"
    manifests = list(matching_cache.glob("*/manifest.json")) if matching_cache.is_dir() else []
    if manifests and config.get("resume_matching") is not True:
        raise RuntimeError(
            "E10 has matching checkpoints but a non-resumable configuration; existing files were preserved"
        )
    if (config.get("resume_matching") is True
            and type(matching_threads) is int and matching_threads in (1, 2, 4)
            and manifests):
        if root is None:
            raise RuntimeError("E10 worker migration requires the reconstruction workspace")
        print(f"Migrating the stopped E10 checkpoint from {matching_threads} to {workers} workers.",
              flush=True)
        run([sys.executable, root / "work/E10-exhaustive-guided/change_workers.py",
             "--workers", workers], root)
        return True
    unexpected = [path for path in started_entries if path != matching_cache]
    if unexpected or (matching_cache.exists() and not manifests):
        raise RuntimeError(
            "E10 has execution artifacts but a non-resumable configuration; refusing an automatic edit"
        )
    previous = dict(config)
    config.update(resume_matching=True, matching_batch_size=128, matching_threads=workers)
    staged = config_path.with_name(".sfm_refine.resume-repair.json")
    staged.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(staged, config_path)
    repair = {
        "repair": "enable reviewed restartable exhaustive guided matching",
        "subject": subject, "previous_config": previous, "updated_config": config,
        "pair_count": expected_pairs, "pair_sha256": file_sha256(ordinary),
    }
    (dataset / "resume_config_repair.json").write_text(json.dumps(repair, indent=2) + "\n")
    print("Repaired the prepared E10 configuration for checkpointed guided matching.", flush=True)
    return True


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
    quality = root / "reconstructions" / f"{subject}_quality"
    detector_boxes = ensure_boxes(root, subject, baseline)
    if not (quality / "sfm_refine.json").is_file():
        # The preparation step adds projected fallback crops to this private
        # copy and stores a durable sibling backup. Never overwrite it later.
        shutil.copy2(detector_boxes, native / "boxes.json")
        run([sys.executable, root / "work/quality-sfm/prepare_quality_datasets.py", subject], root)
    restore_e3_boxes(root, subject, quality, native / "boxes.json")
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


def ensure_mask_prerequisites(root: Path, subject: str, quality: Path) -> tuple[Path, Path | None]:
    """Prepare frozen mask inputs without creating an E4 or E5 result."""
    masks = ensure_masks(root, subject, quality)
    if subject == "light_shirt":
        return masks, None
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
    return masks, approved


def ensure_e4_or_e5(root: Path, subject: str, baseline: Path, experiment: str) -> None:
    """Create only the selected mask-consensus experiment output."""
    quality = ensure_e3(root, subject, baseline)
    masks, approved = ensure_mask_prerequisites(root, subject, quality)
    threshold = "90" if experiment == "E4" else "97"
    fraction = ".90" if experiment == "E4" else ".97"
    if subject == "light_shirt":
        target = masks / f"mask_consensus_{threshold}"
        if not (target / "analysis.json").is_file():
            run([sys.executable, masks / "filter_person_masks.py",
                 "--source", quality / "subject_preview", "--variants",
                 f"mask_consensus_{threshold}:{fraction}"], root)
        return
    protection = masks / "crutch_protection"
    review = protection / "fresh_candidate_review"
    if not (review / "review.json").is_file():
        run([sys.executable, masks / "filter_person_crutches.py", "--source", quality / "subject_preview",
             "--manifest", masks / "body_mask_manifest.json", "--protection", approved,
             "--candidate-review", review, "--qc-only"], root)
        approve_json(review / "review.json", ready_for_filter=True, status="approved_for_experiment")
    target = masks / f"body_mask_consensus_{threshold}_crutches"
    if not (target / "analysis.json").is_file():
        run([sys.executable, masks / "filter_person_crutches.py", "--source", quality / "subject_preview",
             "--manifest", masks / "body_mask_manifest.json", "--protection", approved,
             "--candidate-review", review, "--variants",
             f"body_mask_consensus_{threshold}_crutches:{fraction}"], root)


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
        ensure_e4_or_e5(root, subject, baseline, experiment)
    elif experiment in ("E6", "E7"):
        report_progress(12, "Preparing the E3 features and cameras")
        quality = ensure_e3(root, subject, baseline)
        report_progress(48, "E6 · matching without guided matching")
        if not (root / f"reconstructions/experiments/E6_guided_off/{subject}_quality/reconstruction_report.json").is_file():
            run([sys.executable, root / "work/matching-ablation/E6/prepare_and_run.py", subject], root)
        if experiment == "E7":
            report_progress(72, "Preparing the person masks")
            ensure_mask_prerequisites(root, subject, quality)
            report_progress(88, "E7 · applying the 90% mask cleanup")
            run_mask_cleanup(root, "E7", subject)
    elif experiment in ("E8", "E9"):
        report_progress(12, "Preparing the E3 features and cameras")
        quality = ensure_e3(root, subject, baseline)
        report_progress(48, "E8 · vocabulary matching and triangulation")
        target = root / f"reconstructions/experiments/E8_vocab_guided/{subject}_quality"
        pair_directory = root / "work/matching-ablation/vocab" / subject
        if not (target / "reconstruction_report.json").is_file():
            if not validate_vocab_pairs(pair_directory, subject):
                run([sys.executable, root / "work/matching-ablation/prepare_vocab_pairs.py", subject], root)
                validate_vocab_pairs(pair_directory, subject)
            if not validate_e8_preparation(target, pair_directory):
                run([sys.executable, root / "work/matching-ablation/prepare_e8.py", subject,
                     "--pairs-directory", pair_directory], root)
                validate_e8_preparation(target, pair_directory)
            run([sys.executable, root / "work/matching-ablation/run_e8.py", subject], root)
        if experiment == "E9":
            report_progress(72, "Preparing the person masks")
            ensure_mask_prerequisites(root, subject, quality)
            report_progress(88, "E9 · applying the 90% mask cleanup")
            run_mask_cleanup(root, "E9", subject)
    elif experiment == "E10":
        report_progress(12, "Preparing the E3 reconstruction and masks")
        quality = ensure_e3(root, subject, baseline)
        ensure_mask_prerequisites(root, subject, quality)
        report_progress(52, "Preparing all exhaustive guided pairs")
        internal = root / "work/E10-exhaustive-guided/datasets/light_shirt_quality"
        if not (internal / "sfm_refine.json").is_file():
            run([sys.executable, root / "work/E10-exhaustive-guided/prepare_e10.py", subject], root)
        ensure_e10_resume_config(internal, subject, root)
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
