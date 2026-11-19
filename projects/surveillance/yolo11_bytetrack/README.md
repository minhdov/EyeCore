# In-memory YOLO11 → ByteTrack surveillance

`eyecore run object_tracking` launches `run.py` in the existing YOLO11 environment.
OpenCV reads one frame at a time. A single model calls the official Ultralytics
`model.track(image, tracker="bytetrack.yaml", persist=True)` API; that API performs
both detection and ByteTrack internally in one process, with in-memory detection
objects. The tracker is initialized on the first frame and persists across calls.
This is project-level composition, not communication between two plugin processes.
No YOLO11 or ByteTrack algorithm is implemented here.

Only final `tracks.json` and annotated MJPG `tracks.avi` are written, under
`outputs/object_tracking/in_memory/`. No intermediate detection JSON, crops or
per-frame images are saved. Final track records are accumulated for one JSON export;
video frames themselves are processed and written sequentially.

The existing file-based pipeline is preserved as `eyecore run object_tracking_file`
in `file_based/workflow.yaml`. Standalone `eyecore run yolov11`, `eyecore run bytetrack`
remain supported.

Install the [YOLO11 runtime and pinned weights](../../../plugins/models/yolov11/README.md)
and [ByteTrack requirements](../../../plugins/models/bytetrack/README.md). Both paths
reuse `.cache/yolov11/venv` with Ultralytics **8.3.203** and CPU PyTorch; no package
installation or additional environment is required for Step 4.1.

## Small local video

Use your own short clip or reproduce the validation video from OpenCV's versioned
sample (about 8 MB downloaded, then 30 frames resized to 384×288):

```bash
mkdir -p .cache/surveillance
curl -fL https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/vtest.avi \
  -o .cache/surveillance/vtest.avi
export YOLOV11_PYTHON="$PWD/.cache/yolov11/venv/bin/python"
"$YOLOV11_PYTHON" -I - <<'PY'
import cv2
capture = cv2.VideoCapture('.cache/surveillance/vtest.avi')
assert capture.isOpened()
fps = capture.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter('.cache/surveillance/short.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'), fps, (384, 288))
assert writer.isOpened()
for _ in range(30):
    ok, image = capture.read()
    assert ok
    writer.write(cv2.resize(image, (384, 288)))
capture.release()
writer.release()
PY
```

This produces a three-second real-scene clip at 10 FPS. Downloads and outputs
stay outside Git. Review the upstream sample's rights before redistribution.
The downloaded `vtest.avi` SHA-256 is
`45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`.

## Runtime and workflow

On the configured local installation, from the repository root:

```bash
source .venv/bin/activate
eyecore run object_tracking
```

No module-specific exports are required. EyeCore selects the existing
`.cache/yolov11/venv/bin/python`; the project defaults to the pinned cached YOLO11n
weights, `.cache/surveillance/short.avi`, confidence 0.1, `bytetrack.yaml`, image
size 640 and CPU. Output defaults to `outputs/object_tracking/in_memory/`.
The script owns these algorithm/input defaults; the manifest only selects its
runtime, working directory and execution command.

Optional overrides remain available:

```bash
export YOLOV11_PYTHON=/absolute/path/to/custom-env/bin/python
export YOLOV11_WEIGHTS=/absolute/path/to/custom-model.pt
export SURVEILLANCE_VIDEO=/absolute/path/to/custom-video.avi
```

CLI arguments take precedence over the script's environment/default values.
Unset or empty override variables restore defaults. `YOLO_CONFIG_DIR` optionally
overrides the upstream settings location, which otherwise uses the local ignored
`.cache/yolov11/config/`. If your shell injects unrelated packages, prefix commands
with `env -u PYTHONPATH`. Downloads, environments and outputs remain ignored by Git.

The existing workflow engine launches registered plugins. The small launcher
manifest at `plugins/models/bytetrack/in_memory/plugin.yaml` registers
`object_tracking_pipeline` and points at this project's `run.py`. No core API,
runner behavior or CLI command was added. The `object_tracking` workflow has one
step selecting that launcher.

## Project configuration

`run.py` accepts ordinary command arguments. Change the workflow step's `args`,
run the launcher plugin with extra arguments, or run the script directly:

```bash
.cache/yolov11/venv/bin/python -I projects/surveillance/yolo11_bytetrack/run.py \
  --output /tmp/tracking-results --conf 0.1 --tracker bytetrack.yaml \
  --max-frames 20 --device cpu --imgsz 640

# Extra arguments override the launcher defaults:
eyecore run object_tracking_pipeline -- --max-frames 20 --output /tmp/tracking-results
```

| Argument | Default / meaning |
| --- | --- |
| `--source` | Cached short video; `SURVEILLANCE_VIDEO` overrides |
| `--model` | Cached pinned YOLO11n; `YOLOV11_WEIGHTS` overrides |
| `--output` | `outputs/object_tracking/in_memory/` |
| `--conf` | 0.1; retains low-confidence ByteTrack association candidates |
| `--tracker` | `bytetrack.yaml`, or a local ByteTrack configuration YAML |
| `--max-frames` | All frames; supply a positive integer for a short test |
| `--device` | `cpu`; accepts Ultralytics device syntax, such as `0` for CUDA |
| `--imgsz` | 640 |

The installed runtime has CPU-only PyTorch; GPU execution requires a compatible
user-provided runtime and was not validated. Tracker configs must select
`tracker_type: bytetrack`. The core CLI accepts step options through workflow YAML;
it does not forward extra arguments directly to a workflow.

## Preserved file-based execution

```bash
eyecore run object_tracking_file
```

This retains the original two steps and output paths:

- `outputs/object_tracking/detections/video_detections.json`
- `outputs/object_tracking/tracking/tracks.json`
- `outputs/object_tracking/tracking/tracks.avi`

The in-memory path leaves those files untouched. Both standalone modules still
run independently:

```bash
.cache/yolov11/venv/bin/python -I plugins/models/yolov11/predict.py \
  --source "$PWD/.cache/surveillance/short.avi" --video --conf 0.1 \
  --output /tmp/native-detections
.cache/yolov11/venv/bin/python -I plugins/models/bytetrack/track.py \
  --input /tmp/native-detections/video_detections.json \
  --output /tmp/native-tracking --save-video
```

## Validation and measured comparison

Three alternating sequential CLI trials used the same cached 30-frame, 384×288,
10-FPS clip, YOLO11n weights, confidence 0.1, image size 640, CPU and two threads.
Timing includes CLI/process startup, model loading, inference, tracking and final
JSON/video output. Effective average FPS per run is frames divided by wall time;
the table reports the median across the three runs. This is a short-clip result
on an Intel i9-13900K, not a sustained-throughput benchmark.

The pinned `model.track()` callback initializes ByteTrack with a fixed 30-FPS
frame-rate parameter. The old wrapper uses the source FPS; at 10 FPS its default
`track_buffer: 30` retains lost tracks for 10 frames. For comparable behavior, the
in-memory benchmark used a local copy of `bytetrack.yaml` with `track_buffer: 10`.
Other tracker settings were identical. The standard new workflow still uses the
upstream default configuration, whose lost-track buffer is 30 frames.

| Pipeline | Median total time | Median effective FPS | Peak RSS range | Frames | Track observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| File-based | 3.213 s | 9.34 | 348.3–351.4 MiB | 30 | 195 |
| In-memory | 2.027 s | 14.80 | 361.8–363.6 MiB | 30 | 195 |

These measurements show lower elapsed time for the new path on this clip, with
slightly higher reported peak RSS. RAM was measured with Linux GNU
`/usr/bin/time -f '%M'`: maximum resident size reported for the job, not simultaneous
aggregate memory across its process tree. A JSON report with individual trials
is saved locally at `outputs/step41-benchmark/comparison.json`.

The comparison ran both workflows in a temporary integration root with copied
manifests/project files, protecting existing outputs. To repeat the measurements,
use `/usr/bin/time` around each CLI command, with the compatible tracker YAML
passed in the in-memory workflow step's `args: [--tracker, /absolute/path/to/config.yaml]`.
The preserved file-based project selects its video through its workflow args;
`SURVEILLANCE_VIDEO` is optional there as well. Its confidence 0.1 remains an
explicit project setting, preserving its previous behavior.

Both paths produced matching per-frame IDs/classes/timestamps and 195 observations.
Maximum confidence difference was about 0.000005. Maximum coordinate difference
was 0.155 pixels: the official tracking Results API clips boxes at image bounds,
whereas the file wrapper can return a Kalman-estimated coordinate just outside
(e.g. x2=384.1545 versus 384). The old JSON handoff also rounds detector values to
five decimals. Therefore outputs are not byte-identical.

Tests validate both workflows, final-only artifacts, frame limits, chronological
timestamps and persistent IDs in a deterministic repeated-image video. Existing
synthetic ByteTrack tests cover low-confidence recovery and an empty frame.
Single-image YOLO11, standalone ByteTrack and MuJoCo are regression-checked. Run:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

## Limits

These are execution and continuity smoke tests, not tracking-accuracy results.
There is no ground-truth identity evaluation; IDs can switch, fragment or fail to
initialize. Unconfirmed detections without IDs are omitted from final tracks.
Upstream ByteTrack is class-agnostic and has no appearance re-identification.

Frame IDs start at zero. Decoder timestamps are preserved when available; fallback
FPS estimates are marked `timestamp_estimated: true`. ByteTrack's motion model
uses fixed-step updates, not those timestamps. The AVI is constant-FPS MJPG, and
input dimensions must stay fixed. Final JSON is retained in RAM until export;
this change removes detection serialization, not all memory growth. Existing
files in the chosen final output directory are overwritten. No camera streaming,
multi-camera tracking or distributed execution is added.

The [existing Ultralytics AGPL/Enterprise licensing considerations](../../../plugins/models/bytetrack/README.md#license-and-limitations)
still apply to the official API used here.
