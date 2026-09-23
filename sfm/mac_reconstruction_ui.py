#!/usr/bin/env python3
"""macOS launcher for fresh SfM experiments.

This UI is deliberately separate from the read-only saved-result browser.
It discovers datasets from ``REPOSITORY/datasets/*/images`` and launches the
portable headless engine in a child process so the window remains responsive.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPERIMENTS = tuple(f"E{i}" for i in range(1, 11))
SUBJECT_FOLDERS = {"light_shirt": "light_shirt", "dark_shirt": "black_shirt_crutches"}
DEPENDENCIES = {
    "E1": (),
    "E2": ("E1",),
    "E3": ("E1", "E2"),
    "E4": ("E1", "E2", "E3"),
    "E5": ("E1", "E2", "E3"),
    "E6": ("E1", "E2", "E3"),
    "E7": ("E1", "E2", "E3", "E6"),
    "E8": ("E1", "E2", "E3"),
    "E9": ("E1", "E2", "E3", "E8"),
    "E10": ("E1", "E2", "E3"),
}


def discover_datasets(root: Path = REPOSITORY / "datasets") -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not root.is_dir():
        return result
    for child in sorted(root.iterdir()):
        images = child / "images"
        if child.is_dir() and images.is_dir():
            count = sum(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                        for p in images.rglob("*"))
            if count >= 2:
                result[child.name] = images.resolve()
    return result


def validate_selection(dataset: str, experiment: str, datasets: dict[str, Path]) -> None:
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}")
    if experiment == "E10" and dataset != "light_shirt":
        raise ValueError("E10 is intentionally available only for light_shirt")


def experiment_plan(dataset: str, experiment: str, datasets: dict[str, Path]) -> dict:
    validate_selection(dataset, experiment, datasets)
    return {
        "dataset": dataset,
        "images": str(datasets[dataset]),
        "experiment": experiment,
        "prerequisites": list(DEPENDENCIES[experiment]),
        "output_root": str(data_root() / "sfm" / "mac-e1-e10-workspace" /
                           "reconstructions" / SUBJECT_FOLDERS.get(dataset, dataset)),
    }


def data_root() -> Path:
    value = os.environ.get("CV802_DATA_ROOT")
    if not value:
        raise RuntimeError("Set CV802_DATA_ROOT before launching the reconstruction UI")
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_selection(dataset: str, experiment: str) -> int:
    datasets = discover_datasets()
    plan = experiment_plan(dataset, experiment, datasets)
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "last_plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    command = [sys.executable, str(HERE / "mac_recipe_runner.py"), dataset, experiment]
    print("$ " + " ".join(command), flush=True)
    return subprocess.call(command, cwd=HERE, env=dict(os.environ, CV802_ALLOW_NON_SLURM="1"))


def launch_ui() -> int:
    if sys.platform != "darwin":
        raise RuntimeError("The reconstruction UI is intended for macOS")
    import tkinter as tk
    from tkinter import messagebox, ttk

    datasets = discover_datasets()
    if not datasets:
        raise RuntimeError("No datasets/*/images folders containing images were found")

    window = tk.Tk()
    window.title("CV802 Mac SfM Reconstruction")
    window.geometry("820x560")
    dataset_var = tk.StringVar(value=next(iter(datasets)))
    experiment_var = tk.StringVar(value="E1")
    status_var = tk.StringVar(value="Ready")
    messages: queue.Queue[str | None] = queue.Queue()
    process: list[subprocess.Popen[str] | None] = [None]

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Dataset").grid(row=0, column=0, sticky="w")
    dataset_box = ttk.Combobox(frame, textvariable=dataset_var,
                               values=list(datasets), state="readonly", width=28)
    dataset_box.grid(row=1, column=0, sticky="ew", padx=(0, 12))
    ttk.Label(frame, text="Experiment").grid(row=0, column=1, sticky="w")
    experiment_box = ttk.Combobox(frame, textvariable=experiment_var,
                                  values=EXPERIMENTS, state="readonly", width=12)
    experiment_box.grid(row=1, column=1, sticky="ew")
    log = tk.Text(frame, wrap="word", height=24)
    log.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(14, 0))
    ttk.Label(frame, textvariable=status_var).grid(row=3, column=0, columnspan=3,
                                                   sticky="w", pady=(10, 0))
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(4, weight=1)

    def refresh_experiments(*_args: object) -> None:
        allowed = list(EXPERIMENTS)
        if dataset_var.get() != "light_shirt":
            allowed.remove("E10")
        experiment_box.configure(values=allowed)
        if experiment_var.get() not in allowed:
            experiment_var.set("E9")

    def pump() -> None:
        while True:
            try:
                line = messages.get_nowait()
            except queue.Empty:
                break
            if line is None:
                status_var.set("Finished")
                run_button.configure(state="normal")
            else:
                log.insert("end", line)
                log.see("end")
        window.after(100, pump)

    def worker(command: list[str]) -> None:
        child = subprocess.Popen(command, cwd=HERE, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process[0] = child
        assert child.stdout is not None
        for line in child.stdout:
            messages.put(line)
        code = child.wait()
        messages.put(f"\nProcess exited with status {code}.\n")
        messages.put(None)
        process[0] = None

    def start() -> None:
        try:
            validate_selection(dataset_var.get(), experiment_var.get(), datasets)
            data_root()
        except (ValueError, RuntimeError) as error:
            messagebox.showerror("Cannot start", str(error))
            return
        run_button.configure(state="disabled")
        status_var.set(f"Running {dataset_var.get()} {experiment_var.get()}")
        command = [sys.executable, str(Path(__file__).resolve()), "run",
                   "--dataset", dataset_var.get(), "--experiment", experiment_var.get()]
        threading.Thread(target=worker, args=(command,), daemon=True).start()

    run_button = ttk.Button(frame, text="Recompute", command=start)
    run_button.grid(row=1, column=2, padx=(12, 0))
    dataset_var.trace_add("write", refresh_experiments)
    refresh_experiments()
    window.protocol("WM_DELETE_WINDOW", lambda: window.destroy()
                    if process[0] is None else messagebox.showwarning(
                        "Reconstruction running", "Wait for the current process to finish."))
    window.after(100, pump)
    window.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ui", help="open the macOS reconstruction window")
    commands.add_parser("list", help="list discovered datasets")
    plan = commands.add_parser("plan", help="print an experiment dependency plan")
    run = commands.add_parser("run", help="run a selected experiment")
    for command in (plan, run):
        command.add_argument("--dataset", required=True)
        command.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    args = parser.parse_args(argv)
    if args.command == "ui":
        return launch_ui()
    datasets = discover_datasets()
    if args.command == "list":
        print(json.dumps({name: str(path) for name, path in datasets.items()}, indent=2))
        return 0
    if args.command == "plan":
        print(json.dumps(experiment_plan(args.dataset, args.experiment, datasets), indent=2))
        return 0
    return run_selection(args.dataset, args.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
