"""Run the pinned Ultralytics ByteTrack implementation on detection JSON."""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_detections(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Detection JSON must be an object with fps, width, height and frames")
    if not number(data.get("fps")) or data["fps"] <= 0:
        raise ValueError("fps must be a positive finite number")
    if any(type(data.get(k)) is not int or data[k] <= 0 for k in ("width", "height")):
        raise ValueError("width and height must be positive integers")
    if not isinstance(data.get("frames"), list) or not data["frames"]:
        raise ValueError("frames must be a nonempty list; include frames with no detections")
    previous_time = -1
    for index, frame in enumerate(data["frames"]):
        if not isinstance(frame, dict) or type(frame.get("frame_id")) is not int or frame["frame_id"] != index:
            raise ValueError("frame_id must be contiguous and chronological, starting at 0")
        timestamp = frame.get("timestamp_seconds")
        if not number(timestamp) or timestamp < 0 or timestamp <= previous_time:
            raise ValueError(f"Frame {index}: timestamp_seconds must be finite, nonnegative and increasing")
        previous_time = timestamp
        if not isinstance(frame.get("detections"), list):
            raise ValueError(f"Frame {index}: detections must be a list")
        for detection in frame["detections"]:
            if not isinstance(detection, dict):
                raise ValueError(f"Frame {index}: each detection must be an object")
            if type(detection.get("class")) is not int or detection["class"] < 0:
                raise ValueError(f"Frame {index}: class must be a nonnegative integer")
            confidence = detection.get("confidence")
            if not number(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"Frame {index}: confidence must be in [0, 1]")
            box = detection.get("box")
            if not isinstance(box, dict) or not all(number(box.get(k)) for k in ("x1", "y1", "x2", "y2")):
                raise ValueError(f"Frame {index}: box needs finite pixel coordinates x1, y1, x2, y2")
            if not 0 <= box["x1"] < box["x2"] <= data["width"] or not 0 <= box["y1"] < box["y2"] <= data["height"]:
                raise ValueError(f"Frame {index}: box must have positive area inside the image")
    return data


def track_sequence(data):
    # Disable the library's optional automatic dependency installer. Install requirements first.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[3] / ".cache/yolov11/config"))
    import numpy as np
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import ROOT, YAML

    config = SimpleNamespace(**YAML.load(ROOT / "cfg/trackers/bytetrack.yaml"))
    tracker = BYTETracker(config, frame_rate=data["fps"])
    frames = []
    for frame in data["frames"]:
        rows = [[*[d["box"][k] for k in ("x1", "y1", "x2", "y2")], d["confidence"], d["class"]]
                for d in frame["detections"]]
        boxes = Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), (data["height"], data["width"]))
        tracks = []
        for row in tracker.update(boxes):
            tracks.append({"track_id": int(row[4]), "class": int(row[6]), "confidence": float(row[5]),
                           "box": dict(zip(("x1", "y1", "x2", "y2"), map(float, row[:4])))})
        frames.append({k: v for k, v in frame.items() if k != "detections"} | {"tracks": tracks})
    return {k: v for k, v in data.items() if k != "frames"} | {"frames": frames}


def save_video(data, output):
    import cv2

    source = data.get("source")
    if not isinstance(source, str) or not Path(source).is_file():
        raise ValueError("Annotated video requires an existing local video path in 'source'")
    capture = cv2.VideoCapture(source)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), data["fps"],
                             (data["width"], data["height"]))
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise ValueError("Cannot open source video or MJPG output writer")
        for frame in data["frames"]:
            ok, image = capture.read()
            if not ok or image.shape[:2] != (data["height"], data["width"]):
                raise ValueError(f"Video is missing frame {frame['frame_id']} or has mismatched dimensions")
            for track in frame["tracks"]:
                x1, y1, x2, y2 = [round(track["box"][k]) for k in ("x1", "y1", "x2", "y2")]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"ID {track['track_id']} class {track['class']}", (x1, max(15, y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(image)
    finally:
        capture.release()
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Chronological video_detections.json.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for tracks.json and optional tracks.avi.")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    try:
        data = load_detections(args.input)
        tracked = track_sequence(data)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.save_video:
            save_video(tracked, args.output / "tracks.avi")
        (args.output / "tracks.json").write_text(json.dumps(tracked, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, ImportError) as exc:
        parser.error(f"{exc}. Check the input schema and install requirements in the tracking environment.")
    print(f"Tracked {len(tracked['frames'])} frames; saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
