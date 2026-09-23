"""Plot the two finished sparse point clouds; no generated or filled-in surfaces."""
from pathlib import Path
import json
import numpy as np
import pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
fig, axes = plt.subplots(1, 2, figsize=(9, 8))
fig.subplots_adjust(left=.035, right=.965, bottom=.105, top=.79, wspace=.07)
fig.suptitle("Two separate SfM reconstructions", fontsize=20, weight="bold", y=.975)
for ax, name, label in zip(axes, ("black_shirt_crutches", "light_shirt"),
                           ("Black shirt / crutches", "Light shirt")):
    dataset = root / "reconstructions" / name
    preview = dataset / "subject_preview"
    report = json.loads((dataset / "reconstruction_report.json").read_text())
    analysis = json.loads((preview / "analysis.json").read_text())
    model = pycolmap.Reconstruction(preview / "colmap/sparse/0")
    points = list(model.points3D.values())
    xyz = np.asarray([point.xyz for point in points])
    rgb = np.asarray([point.color for point in points]) / 255.0
    basis = np.asarray(analysis["display_orientation"]["world_to_display_row_matrix"])
    display = (xyz - np.median(xyz, axis=0)) @ basis
    angle = np.deg2rad(15)
    horizontal = display[:, 0] * np.cos(angle) + display[:, 1] * np.sin(angle)
    depth = -display[:, 0] * np.sin(angle) + display[:, 1] * np.cos(angle)
    low, high = np.quantile(display[:, 2], [.005, .995])
    height = high - low
    x = (horizontal - np.median(horizontal)) / height
    up = (display[:, 2] - low) / height
    order = np.argsort(depth)[::-1]  # Draw far points before near points.
    ax.set_facecolor("#e8ebf0")
    ax.scatter(x[order], up[order], c=rgb[order], s=1.3, linewidths=0,
               rasterized=True)
    ax.set(xlim=(-.42, .42), ylim=(-.04, 1.04), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{label}\n{report['registered_images']}/{report['input_images']} views connected"
                 f"\n{len(points):,} person-focused points", fontsize=11, pad=12)
fig.text(.5, .035, "Actual sparse 3D points from your photos and video frames.\n"
         "Background filtered for viewing; each model scaled independently. No MVS or surface mesh.",
         fontsize=9, ha="center", va="bottom")
destination = root / "reconstructions" / "two_subjects_preview.png"
fig.savefig(destination, dpi=180)
plt.close(fig)
print(destination)
