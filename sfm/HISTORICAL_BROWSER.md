# E1–E10 and cross-method saved-result browser and viva guide

## What this feature does

`browse_historical.py` is a read-only catalog, command-line selector and local
WebGL viewer for the preserved historical SfM results plus allow-listed MVS and
VGGSfM results. It offers E1 through E10 and four cross-method groups: raw MVS,
cleaned MVS, raw VGGSfM and cleaned VGGSfM. The subject selector then chooses
Light shirt or Black shirt + crutches when that exact result exists. E10 Black
was not run; cleaned light MVS was not produced. Loading a selection streams
its coloured PLY from the permanent data root; it never starts reconstruction.

This separation is important:

- **inspect preserved result** means open the exact transferred PLY and check its
  recorded vertex count;
- **rerun an experiment** would execute the historical Mac recipe and regenerate
  its inputs/intermediates;
- the first operation is implemented here, while byte-identical Linux rerun
  portability is deliberately **not** claimed;
- E11 is the separate portable fresh-camera server recipe and is not an alias
  for E1–E10.

## Native Open3D desktop window

The browser on port **8765** is the user's preferred interface. It renders
locally in the Mac's browser and avoids remote-desktop frame streaming. The
native Open3D/noVNC window on port 8766 remains an optional alternative; its
server-side software renderer can be noticeably slower while reconstructions
are running.

`historical_desktop.py` provides a desktop view of the same catalog plus
validated cross-method saved results. It is an actual Open3D 0.19 application:
the class subclasses the starter code's original `AppWindow`, uses its native
`SceneWidget`, and retains the original **BG Color** and **Point Size**
controls. It adds the same saved-result dropdown, a subject dropdown, **Load
saved result**, **Reset view**, and **Refresh saved-result availability**. Native
Open3D orbit, pan and wheel zoom replace the browser's bounded camera model, so
body parts can be inspected closely. E10 offers Light shirt only. Raw MVS has
validated Light and Black results with 487,451 and 616,827 points. The separate
`MVSCLEAN / Black` result has 455,181 points; raw MVS remains selectable.
VGGSfM raw has validated Light and Black results with 414,208 and 187,308
points. Their separately labelled `VGGCLEAN` results have 138,677 and 62,889
points. A raw or cleaned PLY is served only while its configured manifest or
publication record, run ID, point count, byte size and SHA-256 match. Derived
cleanup never replaces the corresponding raw result.

The dedicated window does not call COLMAP or any reconstruction engine. Its
COLMAP controls are hidden, arbitrary open/export/fit actions fail closed, and
it reads only the catalog-approved PLY after header/count validation. The
preserved display PLY has colored 3D points but no stored camera frustums, so
the camera color/size controls are disabled rather than inventing camera data.
The upright coordinate transform is display-only and never changes the PLY.

First validate a selection without importing Open3D or opening a window:

```bash
cd /home/anas.khan/cv802_project/project1/ass1/sfm
export CV802_DATA_ROOT=/l/users/anas.khan/cv_802_ass1
python3 -B historical_desktop.py --experiment E10 --subject light --check-only
# The completed dense result uses the same validation-only path:
python3 -B historical_desktop.py --experiment MVS --subject light --check-only
python3 -B historical_desktop.py --experiment VGGSFM --subject light --check-only
```

Inside an active graphical X/VNC session, launch the real desktop window from
the isolated data-root environment:

```bash
export TMPDIR="$CV802_DATA_ROOT/sfm/tmp/open3d-desktop"
export PYTHONPYCACHEPREFIX="$CV802_DATA_ROOT/sfm/cache/pycache"
export XDG_CACHE_HOME="$CV802_DATA_ROOT/sfm/cache/xdg"
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME"
/l/users/anas.khan/cv_802_ass1/sfm/envs/open3d-desktop/bin/python -B \
  historical_desktop.py --experiment E10 --subject light \
  --width 1280 --height 800
```

On the cluster, a native Open3D window needs the project-owned X/VNC display;
the browser viewer's HTTP tunnel alone cannot carry a desktop window. X
authorization, VNC secrets, display logs, screenshots and validation receipts
must remain below `DATA_ROOT/sfm/runtime` or `DATA_ROOT/sfm/logs`, never in the
code root or `/tmp`. The original Mac launcher is not modified by this desktop
entrypoint.

## Start the browser

The browser needs no GPU. Run it only on a cluster host where the data root is
mounted and where cluster policy permits a small local HTTP process. Redirect
Python caches and temporary files before starting:

```bash
cd /home/anas.khan/cv802_project/project1/ass1/sfm
export CV802_DATA_ROOT=/l/users/anas.khan/cv_802_ass1
export TMPDIR="$CV802_DATA_ROOT/sfm/tmp/historical-browser"
export PYTHONPYCACHEPREFIX="$CV802_DATA_ROOT/sfm/cache/pycache"
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX"
python3 -B browse_historical.py serve --bind 127.0.0.1 --port 8765
```

Keep the bind address at `127.0.0.1`. From the computer running your web
browser, create an SSH tunnel to the host running that command:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Then open `http://127.0.0.1:8765`. Drag to rotate, Shift-drag or right-drag to
pan, use the mouse wheel to zoom, and use **Reset view** or double-click to
refit. The camera-distance range was expanded from 1.25–12 to **0.03–120**
normalized scene units; the near plane is 0.002 and the far plane follows
wide zoom-out distances. A normal page refresh loads the change, without
restarting reconstruction or altering any PLY. The viewer is self-contained:
it does not fetch JavaScript, models or assets from a CDN.

The top-right **Background** control offers Dark (the original default), Black,
White, Grey and Custom. Selecting Custom enables the adjacent native colour
picker. The choice changes only WebGL's clear colour and the canvas backing;
the point-colour buffer and PLY RGB values are never changed. The preference is
saved in browser-local storage and survives reset, selection changes, loading a
different cloud and page refresh. If local storage is blocked, the control
still works for the current page and fails safely back to Dark on the next
visit. The toolbar's dark translucent card keeps its labels legible over white
and custom backgrounds and wraps without overlap on a narrow screen.

The current durable server is inside authorized job `265977` on `gpu-04`; that
allocation ends at **23:49:47 Asia/Dubai on 2026-09-19**. A login-node relay
already exposes its loopback-only server at login `127.0.0.1:8765`. From the
Mac, use:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8765:127.0.0.1:8765 anas.khan@10.127.79.236
```

Keep that terminal open and visit `http://127.0.0.1:8765`. This is a two-hop
route (Mac to login relay to compute loopback); it does not expose the viewer
publicly. After the allocation expires, launch a new authorized job and repeat
the compute server/relay setup rather than running against unavailable storage.

For a trackable terminal, run the same server in tmux:

```bash
tmux new-session -s cv802-sfm-browser
# run the exports and server command above, then press Ctrl-b d to detach
tmux attach -t cv802-sfm-browser
```

## Command-line selector

These commands use the same validated catalog as the UI:

```bash
# all 28 selector slots (14 groups times two subjects)
python3 -B browse_historical.py list

# exact method, lineage, subject availability, counts and paths
python3 -B browse_historical.py show E4

# exact preserved display PLY, useful for MeshLab or another future UI
python3 -B browse_historical.py path E8 --subject dark

# check all 26 available files and compare each PLY/header/publication binding
python3 -B browse_historical.py validate

# machine-readable variants
python3 -B browse_historical.py list --json
python3 -B browse_historical.py show E10 --json
python3 -B browse_historical.py validate E3 --subject light --json
```

The production reconstruction root is always resolved below:

```text
/l/users/anas.khan/cv_802_ass1/
└── sfm/historical/transferred/sfm/reconstructions/
```

Catalog paths must be relative, cannot contain `..`, and are checked again
after symlink resolution. A missing file, an escaping symlink, a malformed PLY,
or a vertex-count mismatch fails closed. The HTTP server exposes only four
fixed static assets, the catalog API, validation endpoints and catalog-approved
PLY routes; it is not a general file server.

## E1–E10 choices shown in the UI

| ID | Stage label | Light points | Black points | Displayed result |
| --- | --- | ---: | ---: | --- |
| E1 | Reconstruction | 22,230 | 57,440 | Full baseline sparse scene |
| E2 | Derived cleanup | 7,247 | 15,546 | E1 rectangle-filtered subject |
| E3 | Reconstruction | 25,063 | 46,111 | Higher-detail refinement subject preview |
| E4 | Derived cleanup | 19,519 | 32,997 | E3 mask consensus at 90% |
| E5 | Derived cleanup | 14,812 | 26,790 | E3 mask consensus at 97% |
| E6 | Reconstruction | 22,431 | 40,110 | E3-style refinement with guided matching off |
| E7 | Derived cleanup | 17,614 | 28,429 | E6 mask consensus at 90% |
| E8 | Reconstruction | 24,123 | 43,205 | Vocabulary-pair refinement subject preview |
| E9 | Derived cleanup | 18,781 | 31,097 | E8 mask consensus at 90% |
| E10 | Reconstruction | 21,744 | Not run | Exhaustive/guided refinement, published after 90% cleanup |

“Reconstruction” in this table means the experiment performed matching and
triangulation. It does **not** mean every entry estimated cameras from scratch:
E3, E6, E8 and E10 kept E1 poses and rescaled intrinsics. E10 is categorized as
a reconstruction because it performed exhaustive matching and triangulation;
its only published display PLY is the subsequent 90%-cleaned output. Conversely,
E2, E4, E5, E7 and E9 only filter already-existing 3D points and must not be
described as fresh SfM runs.

## Code map and viva explanation

- `configs/historical_e1_e10.json` is the single source of labels, lineage,
  availability, relative PLY paths and documented point counts.
- `historical_browser/catalog.py` validates the schema, resolves paths below the
  data root, represents a result, and reads PLY headers.
- `historical_browser/cli.py` implements `list`, `show`, `path`, `validate` and
  `serve`; these functions are reusable by a future combined five-method UI.
- `historical_browser/server.py` is a dependency-free read-only HTTP server.
- `historical_browser/web/` contains the local HTML/CSS/WebGL viewer.
- `historical_desktop.py` adapts the original native Open3D `AppWindow` into a
  read-only E1–E10 plus validated raw/clean MVS/VGGSfM saved-result viewer.
- `configs/desktop_saved_views.json` declares cross-method desktop results. A
  subject is enabled only after its exact PLY path and documented count are
  published; the refresh button merely rereads this small config and never
  discovers or accepts an unvalidated output by file existence alone.
- `tests/desktop_smoke.py` is an opt-in real-X/Open3D sweep; it loads catalog
  selections and writes screenshots plus a hash-bearing receipt below
  the data root. It is intentionally not part of headless unit discovery.
- `tests/test_historical_browser.py` tests experiment lineage, E10 availability,
  desktop selector behavior, safe path resolution, symlink and `..` rejection,
  PLY counts, CLI output and HTTP routing.

The browser parses ASCII and binary little/big-endian PLY vertex records. It
requires `x`, `y`, `z`, `red`, `green` and `blue`, recentres the bounding box,
normalizes the largest spatial extent, applies a display-only 180-degree X-axis
rotation for the preserved COLMAP coordinate convention, uploads positions and
colours to WebGL, then draws one circular screen-space point per vertex. Mouse
drag changes yaw and pitch; Shift/right-drag pans; the wheel changes camera
distance. Geometry is not altered on disk.

The main limitation is conceptual, not a hidden implementation gap: these are
sparse point clouds, not dense surfaces, and their differing point counts do
not prove accuracy. E1 also includes background while many later displays are
subject-cleaned, so raw counts are not a fair ranking.

## Verification

Run the focused tests with all mutable paths below the data root:

```bash
cd /home/anas.khan/cv802_project/project1/ass1/sfm
TMPDIR=/l/users/anas.khan/cv_802_ass1/sfm/tmp/historical-browser-tests \
PYTHONPYCACHEPREFIX=/l/users/anas.khan/cv_802_ass1/sfm/cache/pycache \
python3 -B -m unittest -v tests.test_historical_browser
```

The current focused suite passes all 11 unit/API tests, including manifest- and
publication-bound cross-method validation and rejection of incomplete records.

A real headless-Chromium sweep then selected and rendered every one of those 19
clouds with SwiftShader WebGL. Each selection reported the catalog point count,
non-background pixels and WebGL error code zero; rotation/reset passed and the
browser reported no page errors. After the display-only axis correction, E10
light, E4 dark and the compact mobile layout were visually inspected upright.
The machine-readable receipt and screenshots are heavy runtime evidence and
therefore remain at:

```text
/l/users/anas.khan/cv_802_ass1/sfm/logs/browser-checks/20260919T165700Z/
```

The final cross-method browser sweep passed all **26** currently available
clouds: 19 historical SfM, two raw MVS, one cleaned MVS, two raw VGGSfM and two
cleaned VGGSfM results. It exercised wheel input beyond both old limits and
verified rotation/reset plus zero WebGL/page errors. It saved hashed screenshots
for every cross-method selection, including both dark clean results. Its
canonical receipt is
`DATA_ROOT/sfm/logs/browser-checks/final-dark-vgg-clean-bg-v3/receipt.json`.
That receipt also exercises the Dark/Black/White/Grey presets and a custom
`#2457a6` background, reads the exact WebGL clear pixels, proves the loaded
point count and colour-buffer allocation are unchanged, checks persistence
through reset/selection/load/reload, and verifies the compact mobile controls
do not overlap.

The 11 tests above are the browser/catalog tests only; report them separately
from the headless reconstruction and fixed-pose portability suites.

The native desktop was also exercised on the real isolated X display with
Open3D CPU 0.19.0 and llvmpipe. The opt-in sweep created the original
`AppWindow`/native `SceneWidget`, selected all 19 available E1–E10/subject
pairs plus the completed Light MVS result, loaded all 20 PLYs, and compared
each in-memory point count with its catalog. E10 Black remained not-run and MVS
Black remained not-complete. The Open3D build reported no CUDA module, the
validation step had an empty `CUDA_VISIBLE_DEVICES`, and no geometry file was
modified. Four whole-window screenshots were hashed; E4 Black, E10 Light and
the 487,451-point MVS Light view were visually checked upright and rendered.
The canonical receipt is:

```text
/l/users/anas.khan/cv_802_ass1/sfm/logs/desktop/validation/receipt.json
```

That canonical native receipt predates the VGGSfM saved-view addition and
therefore proves the earlier 20-cloud set. VGGSfM Light is proven by the newer
real-browser receipt above; no live native window was restarted merely to
duplicate that evidence.

This real-GUI sweep is runtime validation, not an additional discoverable unit
test; the browser/catalog suite therefore remains 11 tests.
