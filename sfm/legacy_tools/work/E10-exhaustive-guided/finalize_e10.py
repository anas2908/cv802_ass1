#!/usr/bin/env python3
"""Wait optionally, then validate and document the completed LIGHT-only E10.

This never runs matching or reconstruction. --wait-for-pid with no value uses
the main runner PID in status.json. Polling sleeps at most three seconds.
Existing comparison outputs are reused only after their input/code hashes are
verified. Documentation is changed only after every required check passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
DATASET = WORK / "datasets/light_shirt_quality"
FINAL = ROOT / "reconstructions/experiments/E10_quality_exhaustive_guided_consensus90/light_shirt"
E4 = ROOT / "reconstructions/experiments/light_person_mask_cleanup/mask_consensus_90"
ORIGINAL_LOG = WORK / "runs/20260919T132426.759361Z/light_shirt_reconstruction.log"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def wait_for_pid(pid):
    require(pid > 0 and pid != os.getpid(), "Invalid runner PID")
    initial = None
    while True:
        # ps also detects a zombie or a reused PID; neither is the original run.
        process = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "stat=", "-o", "lstart="],
                                 text=True, capture_output=True, check=False)
        line = process.stdout.strip()
        if process.returncode or not line or line.split()[0].startswith("Z"):
            return
        started = " ".join(line.split()[1:])
        if initial is not None and started != initial:
            return
        initial = started
        time.sleep(3)


def matching_configuration_history(config, migrations):
    """Summarize the audited handover chain without losing intermediate trials."""
    def entry(value, **extra):
        workers = value.get("matching_threads", 1)
        batch = value.get("matching_batch_size", 128)
        require(type(workers) is int and workers in (1, 2, 4), "Unsupported E10 matching worker count")
        require(type(batch) is int and batch > 0, "Invalid matching checkpoint batch size")
        return {"matching_workers": workers, "matching_batch_size": batch, **extra}

    if not migrations:
        require(config.get("matching_threads", 1) == 1,
                "A multi-worker resumed run requires preserved migration records")
        return [entry(config)]
    history = [entry(migrations[0]["record"]["config_before"])]
    previous = migrations[0]["record"]["config_before"]
    previous_key = None
    for migration in migrations:
        record = migration["record"]
        require(record["config_before"] == previous, "Migration configuration history is discontinuous")
        if previous_key is not None:
            require(record["source_checkpoint_key"] == previous_key, "Migration checkpoint history is discontinuous")
        previous = record["config_after"]
        previous_key = record["target_checkpoint_key"]
        history.append(entry(previous, migration_record=migration["path"],
                             created_utc=record["created_utc"],
                             completed_pairs_inherited=record["completed_pairs_transferred"]))
    require(previous == config, "Migration history does not end at the final matching configuration")
    return history


def history_sequence(timing, field):
    values = []
    for entry in timing["matching_configuration_history"]:
        if not values or values[-1] != entry[field]:
            values.append(entry[field])
    return " → ".join(map(str, values))


def worker_history_text(timing):
    history = timing["matching_configuration_history"]
    workers = history_sequence(timing, "matching_workers")
    batches = history_sequence(timing, "matching_batch_size")
    if len(history) == 1:
        return (f"The run used {workers} matching worker with checkpoint batches of {batches} pairs. ")
    return (f"The recorded matching-worker sequence was {workers}; the final setting used "
            f"{history[-1]['matching_workers']} workers. Checkpoint batch sizes followed {batches} pairs. "
            "Each handover preserved the previously completed guided pairs. "
            "The total runtime is not a benchmark for any one worker setting. ")


def matching_success_reports(config, model_audit, status):
    """Preserve a successful call report and separately measure the full timeline."""
    decoder = json.JSONDecoder()
    compatible = []
    paths = sorted(set([ORIGINAL_LOG, *WORK.glob("runs/*/light_shirt_reconstruction.log")]))
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        previous_end = 0
        for marker in re.finditer(r"^COMPLETE\s+", text, re.MULTILINE):
            try:
                value, consumed = decoder.raw_decode(text, marker.end())
            except json.JSONDecodeError:
                continue
            section = text[previous_end:marker.start()]
            previous_end = consumed
            if not isinstance(value, dict) or value.get("refinement_recipe") != config:
                continue
            seconds = value.get("elapsed_seconds")
            if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
                    or (ROOT / "reconstructions" / value.get("subject", "")).resolve() != DATASET
                    or value.get("mode") != "fixed_pose_sparse_refinement"
                    or value.get("input_images") != 125 or value.get("registered_images") != 125
                    or value.get("points3D") != model_audit.get("points3D")):
                continue
            compatible.append({"log": str(path), "elapsed_seconds": seconds,
                               "cached_model_load": "SfM: loaded the cached reconstruction." in section,
                               "report": value})
    compute = [item for item in compatible if not item["cached_model_load"]]
    require(compute, "No compatible successful non-cache COMPLETE report found in archived stdout logs; refusing to label cache-load time as compute time")
    selected = max(compute, key=lambda item: item["elapsed_seconds"])
    current = read(DATASET / "reconstruction_report.json")
    def reconstruction_step(step):
        command = step.get("command", [])
        return (step.get("subject") == "light_shirt" and step.get("stage") == "reconstruction"
                and any(Path(str(token)).name == "run_quality.py" for token in command)
                and (ROOT / "reconstructions" / command[-1]).resolve() == DATASET)
    starts = []
    for path in sorted(WORK.glob("runs/*/status.json")):
        archived = read(path)
        for step in archived.get("steps", []):
            if reconstruction_step(step) and step.get("started_utc"):
                starts.append((datetime.fromisoformat(step["started_utc"]), path))
    finishes = [datetime.fromisoformat(step["finished_utc"]) for step in status.get("steps", [])
                if reconstruction_step(step) and step.get("returncode") == 0 and step.get("finished_utc")]
    require(starts and finishes, "Full reconstruction wall timeline is missing from archived/current runner steps")
    first, first_status = min(starts, key=lambda item: item[0])
    last = max(finishes)
    seconds = (last - first).total_seconds()
    require(seconds > 0, "Reconstruction wall timeline is invalid")
    migrations = [{"path": str(path), "sha256": sha(path), "record": read(path)}
                  for path in sorted(WORK.glob("worker_migrations/*.json"))]
    history = matching_configuration_history(config, migrations)
    timing = {
        "actual_reconstruction_seconds": seconds,
        "timing_kind": "Full reconstruction wall timeline, including interruption/worker handover",
        "first_reconstruction_started_utc": first.isoformat(),
        "latest_successful_reconstruction_finished_utc": last.isoformat(),
        "first_start_status_file": str(first_status), "first_start_status_sha256": sha(first_status),
        "latest_successful_call_elapsed_seconds": selected["elapsed_seconds"],
        "source_stdout_log": selected["log"], "source_stdout_log_sha256": sha(Path(selected["log"])),
        "selection": "Preserved call report is the greatest elapsed successful non-cache COMPLETE report compatible with the current recipe, image set and audited full-model point count. Overall time uses the earliest archived reconstruction start through the latest successful reconstruction step finish.",
        "current_internal_report_elapsed_seconds": current.get("elapsed_seconds"),
        "current_internal_report_is_not_used_for_compute_timing": True,
        "final_matching_workers": history[-1]["matching_workers"],
        "final_matching_batch_size": history[-1]["matching_batch_size"],
        "matching_configuration_history": history, "worker_migrations": migrations,
        "compatible_logged_reports": [{k: v for k, v in item.items() if k != "report"} for item in compatible],
        "scope": "Wall time from the first reconstruction start through the last successful reconstruction completion, including earlier interrupted work, worker handover and any intervening pause/cache reopening. Features were already extracted. Later segmentation cleanup and final validation are excluded; this is not CPU time or uninterrupted matching time.",
    }
    return selected["report"], timing


def verified_comparison(path):
    report = read(path)
    require(report.get("self_test") is False and report.get("all_input_hashes_unchanged") is True,
            "Existing comparison is a self-test or lacks successful input verification")
    require(report.get("comparison_script_sha256") == sha(WORK / "compare_e10.py"),
            "Comparison script changed; preserving the existing comparison for review")
    require(len(report.get("subjects", [])) == 1 and report["subjects"][0]["subject"] == "light_shirt",
            "Comparison must contain only the light-shirt subject")
    subject = report["subjects"][0]
    require(subject["cleanup_policy_check"]["passed"] and subject["cleanup_policy_check"]["foreground_agreement"] == .9,
            "Comparison did not verify the same 90% cleanup policy")
    rows = {row["experiment"]: row for row in subject["models"]}
    require(set(rows) == {"E4", "E10"}, "Unexpected comparison inputs")
    watched = dict(report["measurement_helpers"])
    for label, dataset in (("E4", E4), ("E10", FINAL)):
        row = rows[label]
        require(Path(row["dataset"]).resolve() == dataset and row["actual_input_experiment"] == label,
                f"Wrong comparison source for {label}")
        require(row["camera_check_against_E1"]["passed"], f"Camera check failed for {label}")
        watched.update(row["input_model_and_analysis_hashes"])
    for filename, signature in watched.items():
        file = Path(filename)
        require(file.is_file() and file.stat().st_size == signature["bytes"] and sha(file) == signature["sha256"],
                f"Existing comparison input or helper changed: {file}")
    frame = read(Path(subject["E1_frame_analysis"]))
    require(subject["fixed_frame"]["world_to_upright_row_matrix"] == frame["display_orientation"]["world_to_display_row_matrix"]
            and subject["fixed_frame"]["origin_world"] == frame["plot"]["display_origin_world"]
            and subject["fixed_frame"]["selection_upright_min"] == frame["plot"]["upright_bounds"]["min"]
            and subject["fixed_frame"]["selection_upright_max"] == frame["plot"]["upright_bounds"]["max"],
            "E1 comparison frame changed")
    from PIL import Image
    for view in ("front", "side"):
        figure = Path(subject["plots"][view])
        require(figure.resolve().parent == path.parent.resolve(), "Comparison figure points outside its own output folder")
        with Image.open(figure) as image:
            image.verify()
    return report, rows


def duration(seconds):
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours} h {minutes} min {seconds} s" if hours else f"{minutes} min {seconds} s"


def results_document(analysis, rows, comparison, timing, matching):
    first, last = (rows[key]["metrics"] for key in ("E4", "E10"))
    plots = comparison["subjects"][0]["plots"]
    delta = last["retained_comparable_points"] - first["retained_comparable_points"]
    percent = 100 * delta / first["retained_comparable_points"]
    lines = ["# E10 light-shirt results", "",
             f"E10 is complete and validated: **{analysis['retained_points']:,} saved sparse points from the same 125 images**. "
             "Only the final 90%-cleaned light-shirt result is published. The dark E10 run was removed before it started.", "",
             "## Open the saved map", "",
             f"1. Open [run_sfm.command](<{ROOT / 'run_sfm.command'}>).",
             "2. In the starter GUI, choose **File → Open existing result**.",
             f"3. Select [the completed light-shirt folder](<{FINAL}>). The saved point cloud loads from its cache.", "",
             "## Compare with E4", "",
             "E4 and E10 use identical 90% mask-consensus cleanup. The comparison also fixes the E1 camera frame, bounds and pixel units; "
             "comparable points need at least three distinct views, mean error ≤ 3 E1 pixels and maximum acute triangulation angle ≥ 1.5°.", "",
             "| Model | Saved cleaned points | Comparable points | Mean error (E1 px) | Occupied voxels (height/100) |",
             "|---|---:|---:|---:|---:|"]
    for label in ("E4", "E10"):
        m = rows[label]["metrics"]
        lines.append(f"| {label} | {m['input_subject_points']:,} | {m['retained_comparable_points']:,} | "
                     f"{m['baseline_pixel_mean_point_error_retained']['mean']:.4f} | {m['occupied_voxels']['height/100']:,} |")
    lines += ["", f"E10 has {delta:+,} comparable points ({percent:+.2f}%) relative to E4. "
              "This describes sparse coverage, not anatomical accuracy; more points can include noise. "
              "These automated checks do not establish a visual improvement or a best reconstruction.", "",
              f"[Front comparison](<{plots['front']}>) · [Side comparison](<{plots['side']}>) · "
              f"[Detailed measurements](<{WORK / 'comparison/comparison.json'}>)", "",
              "## Method and measured time", "",
              "The same E3 native images and saved affine/DSP SIFT features were reused. All **7,750 unique image pairs** "
              "were submitted to COLMAP with guided matching enabled on CPU. "
              + worker_history_text(timing) +
              "Guidance can operate only when initial geometric verification supplies suitable geometry. "
              "E1 camera poses stayed fixed, intrinsics were scaled to native resolution, and the same E3 triangulation settings were used. "
              "The existing segmentation masks then applied the pooled-view 90% consensus rule; no 97% variant was added.", "",
              f"Full reconstruction wall time: **{duration(timing['actual_reconstruction_seconds'])}** "
              f"({timing['actual_reconstruction_seconds']:.3f} seconds), measured from the first reconstruction step's start "
              "through the latest successful reconstruction step's finish. This includes the earlier work, all worker handovers "
              "and intervening pauses or cache reopening, while excluding prior feature extraction, subsequent cleanup and final validation. "
              f"The preserved successful resumed call alone reports {duration(timing['latest_successful_call_elapsed_seconds'])}; "
              "that per-call time is recorded separately and is not presented as the full E10 runtime.", "",
              f"[Preserved compute report](<{WORK / 'measured_reconstruction_report.json'}>) · "
              f"[Timing provenance](<{WORK / 'reconstruction_timing.json'}>)", "",
              "## Validation", "",
              f"Both final matching tables contain all {matching['expected_unique_pairs']:,} requested pair IDs, including unsuccessful attempts. "
              "The matching records agree with the durable checkpoint, feature bytes agree with E3, geometry and camera checks passed, "
              "and the cleaned result opens without extraction, matching or triangulation. "
              "The 90% value describes foreground-mask agreement, not reconstruction accuracy.", "",
              f"[Matching audit](<{WORK / 'final_matching_audit.json'}>) · "
              f"[Final cache check](<{WORK / 'final_cache_verification.json'}>) · "
              f"[Finalization record](<{WORK / 'finalization.json'}>)", ""]
    return "\n".join(lines)


def documentation_updates(points, timing):
    """Prepare bounded status/table edits; a changed marker fails safely."""
    edits = {}
    result_link = f"[E10 results and opening guide](<{ROOT / 'E10_RESULTS.md'}>)"

    def replace(path, old, new):
        before, text = edits.get(path, (path.read_text(), path.read_text()))
        if old in text:
            require(text.count(old) == 1, f"Ambiguous E10 status marker in {path}")
            text = text.replace(old, new, 1)
        else:
            require(new in text, f"Cannot safely update changed E10 status marker in {path}")
        edits[path] = (before, text)

    def replace_pattern(path, pattern, new):
        before, text = edits.get(path, (path.read_text(), path.read_text()))
        updated, count = re.subn(pattern, lambda _: new, text, flags=re.DOTALL)
        require(count == 1 or (count == 0 and new in text), f"Ambiguous or changed E10 paragraph in {path}")
        edits[path] = (before, updated)

    table = ROOT / "E1_E10_COMPARISON.md"
    text = table.read_text()
    lines = text.splitlines(keepends=True)
    found = 0
    for index, line in enumerate(lines):
        if line.startswith("| **E10 —"):
            cells = line.rstrip("\n").split("|")
            require(len(cells) == 12, "Unexpected E10 comparison-table columns")
            cells[4] = " **7,750 completed / —** "
            cells[7] = (f" CPU; **{history_sequence(timing, 'matching_workers')} matching workers** during the run; "
                        f"**{timing['final_matching_workers']} final**; "
                        f"{history_sequence(timing, 'matching_batch_size')} pairs per checkpoint batch; "
                        "not a single-setting timing benchmark ")
            cells[10] = f" **{points:,} / —** "
            lines[index] = "|".join(cells) + "\n"
            found += 1
    require(found == 1, "E10 comparison-table row is missing or ambiguous")
    edits[table] = (text, "".join(lines))
    replace(table, "**E10 runs only the light-shirt subject**", "**E10 is complete for the light-shirt subject only**")
    replace(table, "Among the completed 90%-cleaned results, E4 has the most retained points. That indicates sparse coverage, not proven anatomical accuracy. E10 has no measured quality result yet.",
            f"The completed E10 light-shirt result has {points:,} cleaned points. See {result_link} for its like-for-like E4 comparison and measured runtime. These counts describe sparse coverage, not proven anatomical accuracy.")
    replace(table, "Open `E10_progress.command` in the SfM folder for live progress.", "E10 final validation has completed; the saved result is ready to open.")
    replace(ROOT / "README.md", "**E10 is running for light shirt only**: high-detail exhaustive\nmatching with guided refinement, followed by one 90%-cleaned result. Double-click\n`E10_progress.command` to follow its live progress in Terminal.",
            f"**E10 is complete for light shirt only**, with **{points:,} points** in its\n90%-cleaned exhaustive/guided result. Use the {result_link} for the saved map,\ncomparison figures and measured compute time.")
    replace(ROOT / "EXPERIMENTS.md", "E10 is still running; do not claim\nan improvement before comparing the resulting cleaned maps.",
            f"E10 is complete and validated: **{points:,} cleaned light-shirt points**.\nSee {result_link} for the E4 comparison and measured runtime. Point counts alone\ndo not establish visual improvement or anatomical accuracy.")
    replace(ROOT / "EXPERIMENTS.md", "| Light shirt only; running |", f"| Light shirt only; complete, {points:,} cleaned points |")
    replace(ROOT / "EXPERIMENTS.md", "The long run uses opt-in checkpoints", "The completed run used opt-in checkpoints")
    replace(ROOT / "EXPERIMENTS.md", "experiments and source photographs remain separate. Open `E10_progress.command`\nfor a read-only live Terminal display; closing that display does not stop SfM.",
            "experiments and source photographs remain separate. E10 matching, cleanup and\nfinal verification are complete; the result is available through the starter GUI.")
    replace(ROOT / "reconstructions/experiments/README.md", "**E10 is running for light shirt only**, with high-detail exhaustive pairing,\nguided matching and a single consensus-90 result. Its final destination is\n`E10_quality_exhaustive_guided_consensus90/light_shirt`. See the\n[E1–E10 comparison table](../../E1_E10_COMPARISON.md); `E10_progress.command` in the\nSfM root opens its live Terminal progress display.",
            f"**E10 is complete for light shirt only**, with **{points:,} points** after\nhigh-detail exhaustive/guided matching and the established consensus-90 cleanup.\nOpen `E10_quality_exhaustive_guided_consensus90/light_shirt` through the starter\nGUI's **File → Open existing result**. See the {result_link}.")
    replace(WORK / "README.md", "The current matching and intermediate models are internal to", "The completed matching and intermediate models remain internal to")
    replace_pattern(WORK / "README.md", r"Run `\.\./\.\./E10_progress\.command` to see the read-only Terminal dashboard\. It reports durably saved pair counts every batch of \d+\. Closing that display does not stop reconstruction\.",
            "The read-only `../../E10_progress.command` dashboard can still show the completed pair counts.")
    replace_pattern(WORK / "README.md", r"Do not start another runner while the current one owns `runner\.lock`\.[^\n]*(?:\n(?!\n)[^\n]*)*",
                    "The light-only runner has completed. Archived step timestamps retain the full reconstruction wall timeline, including the worker handover; successful stdout reports retain individual call timings. Logs remain under `runs/`, and `status.json` identifies the completed invocation.")
    replace(WORK / "README.md", "After the reconstruction and 90% cleanup finish, final matching can be checked with `audit_e10_matching.py` and the cleaned model compared to E4 with `compare_e10.py`.",
            "Final matching was checked with `audit_e10_matching.py`, the cleaned model was compared to E4 with `compare_e10.py`, and its GUI cache was verified.")
    replace(WORK / "README.md", "Result counts and measured runtime remain pending until the actual run finishes.",
            f"E10 is complete with **{points:,} cleaned points**. {worker_history_text(timing)}The {result_link} contains actual measurements and the full reconstruction wall timeline.")
    return edits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pid", nargs="?", type=int, const=0,
                        help="Wait for PID, or use the main runner PID from status.json when no PID is supplied")
    args = parser.parse_args()
    record = {"experiment": "E10", "subject": "light_shirt", "started_utc": now(),
              "pid": os.getpid(), "complete": False, "steps": []}
    lock_file = WORK / "finalizer.lock"
    with lock_file.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another E10 finalizer already owns finalizer.lock; its status was preserved.", file=sys.stderr)
            return 2
        try:
            if args.wait_for_pid is not None:
                pid = args.wait_for_pid or read(WORK / "status.json")["pid"]
                record.update(stage="waiting_for_light_runner", waiting_for_pid=pid)
                save(WORK / "finalization.json", record)
                print(f"Waiting for light runner PID {pid}; no matching or database audit is started.", flush=True)
                wait_for_pid(pid)
            status = read(WORK / "status.json")
            require(status.get("complete") is True and status.get("subjects") == ["light_shirt"]
                    and status.get("stage") != "failed", "Latest runner did not finish successfully for light shirt only")
            require((FINAL / "analysis.json").is_file(), "Final light consensus90 result is missing")
            analysis = read(FINAL / "analysis.json")
            require(analysis.get("registered_images") == 125 and analysis.get("filter", {}).get("foreground_agreement") == .9,
                    "Final result does not match the 125-image consensus90 scope")
            run = Path(status["run_directory"]).resolve()
            require(run.is_relative_to(WORK / "runs"), "Runner report is outside the E10 run directory")
            model_step = next((s for s in reversed(status["steps"]) if s.get("subject") == "light_shirt"
                               and s.get("stage") == "model_audit" and s.get("returncode") == 0), None)
            require(model_step is not None, "Latest runner has no successful light model audit")
            command = model_step["command"]
            model_audit_path = Path(command[command.index("--output") + 1]).resolve()
            require(model_audit_path.is_relative_to(run), "Model audit does not belong to the latest runner")
            model_audit = read(model_audit_path)
            require(model_audit.get("passed") is True and model_audit.get("registered_images") == 125,
                    "Latest light model audit failed")
            own_run = WORK / "finalization_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            own_run.mkdir(parents=True)
            record.update(stage="validating", runner_status=status, runner_model_audit=str(model_audit_path),
                          runner_model_audit_sha256=sha(model_audit_path), run_directory=str(own_run))
            save(WORK / "finalization.json", record)

            def step(label, command):
                entry = {"stage": label, "command": list(map(str, command)), "started_utc": now(),
                         "log": str(own_run / (label + ".log"))}
                record["steps"].append(entry)
                record["stage"] = label
                save(WORK / "finalization.json", record)
                with Path(entry["log"]).open("x") as output:
                    result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
                entry.update(returncode=result.returncode, finished_utc=now())
                save(WORK / "finalization.json", record)
                require(result.returncode == 0, f"{label} failed; see {entry['log']}")

            # A new runner must not start while reports are verified/published.
            with (WORK / "runner.lock").open("rb") as runner_lock:
                fcntl.flock(runner_lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                require(read(WORK / "status.json") == status, "Runner status changed before finalization")
                step("final_matching_audit", [sys.executable, str(WORK / "audit_e10_matching.py"), str(DATASET),
                                               "--output", str(WORK / "final_matching_audit.json")])
                matching = read(WORK / "final_matching_audit.json")
                require(matching.get("passed") is True and matching.get("image_count") == 125
                        and matching.get("expected_unique_pairs") == 7750, "Final exhaustive matching proof failed")
                comparison_path = WORK / "comparison/comparison.json"
                if not comparison_path.parent.exists():
                    step("comparison", [sys.executable, str(WORK / "compare_e10.py"), "--subjects", "light_shirt",
                                        "--output-dir", str(comparison_path.parent)])
                comparison, rows = verified_comparison(comparison_path)
                record["comparison_verified"] = True
                require(rows["E10"]["metrics"]["input_subject_points"] == analysis["retained_points"],
                        "Comparison and final cleanup point counts differ")
                step("final_cache_verification", [sys.executable, str(ROOT / "work/quality-sfm/verify_cached_results.py"),
                                                   str(FINAL), "--output", str(WORK / "final_cache_verification.json")])
                cached = read(WORK / "final_cache_verification.json")
                require(cached.get("all_cached") is True and cached.get("forbidden_operations_called") == []
                        and len(cached.get("datasets", [])) == 1
                        and cached["datasets"][0].get("canonical_files_unchanged") is True
                        and cached["datasets"][0].get("displayed_points") == analysis["retained_points"],
                        "Final cleaned model did not open from its unchanged cache")
                measured, timing = matching_success_reports(read(DATASET / "sfm_refine.json"), model_audit, status)
                measured_path = WORK / "measured_reconstruction_report.json"
                if measured_path.exists():
                    require(read(measured_path) == measured, "Preserving a different existing measured reconstruction report")
                else:
                    save(measured_path, measured)
                save(WORK / "reconstruction_timing.json", timing)
                document = results_document(analysis, rows, comparison, timing, matching)
                result_path = ROOT / "E10_RESULTS.md"
                require(not result_path.exists() or result_path.read_text() == document,
                        "Preserving an existing edited E10_RESULTS.md; review before replacing it")
                edits = documentation_updates(analysis["retained_points"], timing)
                record.update(stage="checks_passed_publishing", all_required_checks_passed=True,
                              actual_reconstruction_seconds=timing["actual_reconstruction_seconds"],
                              saved_cleaned_points=analysis["retained_points"],
                              E4_saved_points=rows["E4"]["metrics"]["input_subject_points"])
                save(WORK / "finalization.json", record)
                # Build and verify all bounded edits first; then publish. Resume
                # recognizes both original and already-updated status markers.
                require(all(path.read_text() == original for path, (original, _) in edits.items()),
                        "Documentation changed while preparing the final status updates")
                atomic_text(result_path, document)
                for path, (_, updated) in edits.items():
                    atomic_text(path, updated)
                outputs = [result_path, measured_path, WORK / "reconstruction_timing.json",
                           WORK / "final_matching_audit.json", WORK / "final_cache_verification.json", comparison_path,
                           *[Path(p) for p in comparison["subjects"][0]["plots"].values()], *edits]
                record.update(stage="complete", complete=True, finished_utc=now(),
                              artifacts_sha256={str(path): sha(path) for path in outputs})
                save(WORK / "finalization.json", record)
                save(own_run / "finalization.json", record)
                print(json.dumps({"complete": True, "saved_points": analysis["retained_points"],
                                  "reconstruction_wall_seconds": timing["actual_reconstruction_seconds"],
                                  "results": str(result_path)}, indent=2), flush=True)
                return 0
        except BaseException as error:
            record.update(stage="failed", complete=False, error_type=type(error).__name__, error=str(error), finished_utc=now())
            save(WORK / "finalization.json", record)
            print(json.dumps({"complete": False, "error": str(error), "status": str(WORK / 'finalization.json')}, indent=2), file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
