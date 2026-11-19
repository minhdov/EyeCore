# ByteTrack plugin

Uses `ultralytics.trackers.byte_tracker.BYTETracker` from **Ultralytics 8.3.203**.
The wrapper converts detection JSON to Ultralytics `Boxes`, loads the upstream
`bytetrack.yaml` defaults and calls one tracker instance once per frame, including
empty frames. Matching and motion estimation remain entirely in the library.

## Environment

Reuse the existing CPU YOLO11 environment; no additional PyTorch installation:

```bash
export YOLOV11_PYTHON="$PWD/.cache/yolov11/venv/bin/python"
"$YOLOV11_PYTHON" -m pip install -r plugins/models/bytetrack/requirements.txt
export YOLO_CONFIG_DIR="$PWD/.cache/yolov11/config"
```

The only new dependency is `lap==0.5.12` for upstream assignment. Nothing is
installed into EyeCore's `.venv`. Automatic dependency installation is disabled
in the wrapper; install the requirements explicitly before use.

## Standalone smoke test

Create a small sequence with the same object in consecutive frames:

```bash
cat > /tmp/bytetrack-detections.json <<'JSON'
{
  "fps": 10, "width": 200, "height": 150,
  "frames": [
    {"frame_id": 0, "timestamp_seconds": 0.0, "detections": [
      {"class": 0, "confidence": 0.9, "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 70}}
    ]},
    {"frame_id": 1, "timestamp_seconds": 0.1, "detections": [
      {"class": 0, "confidence": 0.9, "box": {"x1": 12, "y1": 10, "x2": 42, "y2": 70}}
    ]}
  ]
}
JSON
"$YOLOV11_PYTHON" -I plugins/models/bytetrack/track.py \
  --input /tmp/bytetrack-detections.json --output /tmp/bytetrack-results
```

No EyeCore installation is required in this environment. `tracks.json` preserves
frame IDs and timestamps and replaces `detections` with `tracks`. Every track has
an integer `track_id`, integer `class`, `confidence`, and a pixel-coordinate `box`.
The same ID persists across both frames in this smoke test.

Input IDs must be contiguous and zero-based, timestamps finite and strictly
increasing, and FPS/dimensions positive. Confidence must be in [0, 1], class IDs
nonnegative integers and boxes positive-area pixel rectangles inside the image.
Include frames without detections as `"detections": []`; frames are never silently
sorted or skipped. Invalid/missing files produce actionable errors before tracking.

Add an absolute local video path in `source` and pass `--save-video` to produce
`tracks.avi` using MJPG. The video must correspond to the sequence and dimensions.
It is decoded chronologically and labeled with confirmed track IDs. The AVI uses
constant FPS; JSON preserves the original timestamps. Output files are overwritten.

## EyeCore

The manifest automatically shares the installed `.cache/yolov11/venv` runtime;
`YOLOV11_PYTHON` optionally overrides it. No export is required. The existing
surveillance handoff paths remain unchanged, and tracker settings continue to
come directly from the pinned library's `bytetrack.yaml`:

```bash
source .venv/bin/activate
eyecore info bytetrack
eyecore run bytetrack -- --input /tmp/bytetrack-detections.json --output /tmp/bytetrack-results
```

For detection and tracking composed in memory, see the
[object_tracking project](../../../projects/surveillance/yolo11_bytetrack/README.md).
Clear inherited Python packages with `env -u PYTHONPATH` if needed.

## License and limitations

This is the **Ultralytics implementation**, with its
[AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/v8.3.203/LICENSE),
not a claim that this integration inherits another ByteTrack repository's license.
The existing [Ultralytics commercial licensing considerations](../yolov11/README.md#license-and-commercial-distribution)
apply. [LAP's license](https://github.com/gatagat/lap/blob/v0.5.12/LICENSE) is
BSD-2-Clause. Resolve source-disclosure obligations or Enterprise terms before
proprietary distribution; process isolation is not a license exemption.

The direct tracker API is tied to the pinned version. IDs reset per invocation.
ByteTrack uses IoU/motion association without appearance re-identification and
is class-agnostic; occlusions, crossings or class changes can cause ID switches.
Empty frames advance state but do not emit predicted/lost tracks. FPS drives
buffer aging; timestamps are metadata, not a variable-time motion model. Full
sequences are held in memory. This offline CPU example is not an accuracy
benchmark, real-time camera service or multi-camera system.
