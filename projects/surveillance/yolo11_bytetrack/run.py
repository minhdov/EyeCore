"""YOLO11 detection and persistent ByteTrack tracking in one process."""

import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--source", type=Path,
                        default=Path(os.environ.get("SURVEILLANCE_VIDEO") or root / ".cache/surveillance/short.avi"))
    parser.add_argument("--model", type=Path,
                        default=Path(os.environ.get("YOLOV11_WEIGHTS") or root / ".cache/yolov11/assets/yolo11n.pt"))
    parser.add_argument("--output", type=Path, default=root / "outputs/object_tracking/in_memory")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--tracker", default="bytetrack.yaml", help="ByteTrack YAML name or local path.")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0 for CUDA.")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not 0 < args.conf <= 1 or args.imgsz <= 0:
        parser.error("--conf must be in (0, 1] and --imgsz must be positive")
    for name, path in (("source", args.source), ("model", args.model)):
        if not path.is_file():
            parser.error(f"{name} file does not exist: {path}")

    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache/yolov11/config"))
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    capture = cv2.VideoCapture(str(args.source))
    writer = None
    try:
        config = YAML.load(check_yaml(args.tracker))
        if config.get("tracker_type") != "bytetrack":
            raise ValueError("--tracker must select tracker_type: bytetrack")
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {args.source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video needs valid FPS; convert it to a constant-frame-rate clip first")
        model = YOLO(str(args.model), task="detect")
        frames = []
        width = height = 0
        while args.max_frames is None or len(frames) < args.max_frames:
            ok, image = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = image.shape[:2]
                args.output.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.output / "tracks.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise ValueError("Cannot open MJPG output writer")
            elif image.shape[:2] != (height, width):
                raise ValueError("Video dimensions changed during decoding")
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            estimated = not math.isfinite(timestamp) or timestamp < 0 or bool(
                frames and timestamp <= frames[-1]["timestamp_seconds"])
            if estimated:
                timestamp = max(len(frames) / fps, frames[-1]["timestamp_seconds"] + 1 / fps) if frames else 0

            # The official API performs detection and tracking internally, in memory.
            # persist=True initializes the tracker once and retains it across frames.
            result = model.track(image, tracker=args.tracker, persist=True, device=args.device,
                                 imgsz=args.imgsz, conf=args.conf, save=False, verbose=False)[0]
            tracks = []
            if result.boxes.is_track:
                for x1, y1, x2, y2, track_id, confidence, class_id in result.boxes.data.cpu().numpy():
                    track = {"track_id": int(track_id), "class": int(class_id), "confidence": float(confidence),
                             "box": dict(zip(("x1", "y1", "x2", "y2"), map(float, (x1, y1, x2, y2))))}
                    tracks.append(track)
                    cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
                    cv2.putText(image, f"ID {int(track_id)} class {int(class_id)}", (round(x1), max(15, round(y1))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(image)
            frames.append({"frame_id": len(frames), "timestamp_seconds": timestamp,
                           "timestamp_estimated": estimated, "tracks": tracks})
        if not frames:
            raise ValueError("Video contains no readable frames")
        payload = {"source": str(args.source.resolve()), "fps": fps, "width": width, "height": height,
                   "frames": frames}
        (args.output / "tracks.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Tracked {len(frames)} frames in memory; saved {args.output.resolve()}")
    except (OSError, ValueError, AssertionError) as exc:
        parser.error(str(exc))
    finally:
        capture.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
