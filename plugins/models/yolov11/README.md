# YOLO11 detection plugin

Runs the official Ultralytics YOLO11n detector on one local image. The small
`predict.py` wrapper calls `YOLO.predict()`, `Results.save()` and `Results.to_json()`
to save both an annotated image and JSON. It imports no EyeCore code. Native CLI
inference is also shown below; JSON export uses the official Python Results API.

Validated on Linux x86_64 with Python 3.11.14, Ultralytics 8.3.203, CPU-only
PyTorch 2.6.0+cpu and torchvision 0.21.0+cpu. No training or GPU is required.
The pretrained `yolo11n.pt` is pinned to Ultralytics assets release **v8.3.0**.
The sample image is pinned to the Ultralytics **v8.3.203** source tag. Both hashes
are recorded in `assets.sha256`; weights and images are downloaded, not committed.

## Installation

From the EyeCore repository root, create an independent runtime. Never install
these requirements into EyeCore's main `.venv`:

```bash
.venv/bin/python -m venv .cache/yolov11/venv
.cache/yolov11/venv/bin/python -m pip install \
  torch==2.6.0+cpu torchvision==0.21.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
.cache/yolov11/venv/bin/python -m pip install \
  -r plugins/models/yolov11/requirements.txt

mkdir -p .cache/yolov11/assets .cache/yolov11/config
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt \
  -o .cache/yolov11/assets/yolo11n.pt
curl -fL https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.203/ultralytics/assets/bus.jpg \
  -o .cache/yolov11/assets/bus.jpg
(cd .cache/yolov11/assets && sha256sum -c ../../../plugins/models/yolov11/assets.sha256)
```

Install CPU PyTorch first so the second command reuses those pinned builds instead
of downloading CUDA dependencies. `requirements.txt` pins the primary packages;
transitive dependencies are not fully locked. The core still depends only on
Typer and PyYAML. `.cache/` and `outputs/` are already ignored by Git.

## Runtime configuration

After installation, no exports are required for `eyecore run yolov11`. The
manifest selects `.cache/yolov11/venv/bin/python` automatically. The module itself
defaults to the cached pinned `yolo11n.pt`, cached `bus.jpg`, confidence 0.25,
image size 640 and `outputs/yolov11/`. EyeCore does not define those algorithm
defaults. Optional overrides for an external runtime or custom assets:

```bash
export YOLOV11_PYTHON=/absolute/path/to/custom-env/bin/python
export YOLOV11_WEIGHTS=/absolute/path/to/custom-model.pt
export YOLOV11_SOURCE=/absolute/path/to/image.jpg
```

Omitting or emptying these variables restores the defaults. Explicit `--model`
and `--source` arguments override the module's environment/default values.
Ultralytics settings default to `.cache/yolov11/config/`; optional `YOLO_CONFIG_DIR`
overrides that upstream location. If your shell has sourced ROS2 or another Python
environment, prefix commands with `env -u PYTHONPATH` to avoid mixing packages.
The runner uses the selected interpreter and does not activate environments.

## Independent execution

The official native command works without EyeCore and saves the annotated image:

```bash
.cache/yolov11/venv/bin/yolo detect predict \
  model="$PWD/.cache/yolov11/assets/yolo11n.pt" source="$PWD/.cache/yolov11/assets/bus.jpg" \
  device=cpu imgsz=640 conf=0.25 save=True \
  project="$PWD/outputs/yolov11-validation" name=native exist_ok=True
```

The standalone wrapper saves the image and JSON, also without EyeCore installed
in its environment. `-I` additionally isolates Python from the working directory
and Python environment variables:

```bash
.cache/yolov11/venv/bin/python -I plugins/models/yolov11/predict.py \
  --output "$PWD/outputs/yolov11-validation/independent"
```

## EyeCore execution

Activate the core environment for the CLI; module exports are optional:

```bash
source .venv/bin/activate
eyecore list
eyecore info yolov11
eyecore run yolov11
```

The output files are `outputs/yolov11/annotated.jpg` and
`outputs/yolov11/detections.json`. JSON is the official Ultralytics list of
detections: each includes `name`, `class`, `confidence`, and `box` with pixel
coordinates `x1`, `y1`, `x2`, `y2` (not normalized). An image without detections
produces an empty list. Files in the selected output directory are overwritten.

Run on your own local image or choose a different output directory:

```bash
eyecore run yolov11 -- --source /absolute/path/to/image.jpg \
  --output /absolute/path/to/results --conf 0.4 --imgsz 640
```

Appended arguments override the module's defaults and environment values.
Missing files fail before inference; the wrapper
does not automatically download missing weights. Single-image mode is unchanged.
Add `--video` to export chronological `video_detections.json` with frame IDs,
timestamps, dimensions and FPS; `--max-frames N` optionally limits the sequence.
This performs detection only. See the separate
[ByteTrack surveillance stage](../../../projects/surveillance/yolo11_bytetrack/README.md).

## Validation results

Native CLI, independent wrapper and EyeCore execution each exited 0 and detected
**four people and one bus** at confidence 0.25 and image size 640. The standalone
and EyeCore JSON outputs matched, and both annotated images were saved at the
original 810 x 1080 dimensions. Core file hashes and the original modules,
projects and generated outputs were checked for preservation. Core regression
tests run without installing Ultralytics in the core environment; optional
integration tests use the independent environment when installed.

## License and commercial distribution

Ultralytics publishes its software under
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/v8.3.203/LICENSE)
and offers an [Enterprise License](https://www.ultralytics.com/license).
Its licensing guidance applies to its code and models, including pretrained
weights. The vendor states that integrating applications must either comply with
AGPL source-disclosure requirements or obtain an Enterprise License; proprietary
deployment requires the Enterprise option under that guidance.

Before commercial distribution or deployment, review the applicable AGPL
obligations for corresponding source, license/copyright notices and network
interaction, or arrange the appropriate Enterprise terms with Ultralytics.
Running YOLO in a separate process or environment does not establish an exemption.
Commercial use is not categorically prohibited by AGPL, but proprietary use must
not assume that a permissive license on another component overrides these terms.
This plugin does not grant an Enterprise License or relicense EyeCore. Model
weights and the upstream demonstration image remain local evaluation assets;
review their rights before redistributing them.

Official references: [YOLO11](https://docs.ultralytics.com/models/yolo11/),
[prediction/Results API](https://docs.ultralytics.com/modes/predict/),
[CPU PyTorch wheels](https://docs.pytorch.org/get-started/previous-versions/).
