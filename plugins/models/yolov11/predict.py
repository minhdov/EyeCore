"""Detect objects in a local image or video using the official Ultralytics API."""

import argparse
import json
import math
import os
from pathlib import Path


def detect_video(model, args):
    import cv2

    capture = cv2.VideoCapture(str(args.source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {args.source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS; convert it to a constant-frame-rate clip first")
        frames = []
        width = height = 0
        while args.max_frames is None or len(frames) < args.max_frames:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            frame_id = len(frames)
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            estimated = not math.isfinite(timestamp) or timestamp < 0 or (
                frames and timestamp <= frames[-1]["timestamp_seconds"]
            )
            if estimated:
                timestamp = max(frame_id / fps, frames[-1]["timestamp_seconds"] + 1 / fps) if frames else 0
            result = model.predict(image, device="cpu", imgsz=args.imgsz, conf=args.conf,
                                   save=False, verbose=False)[0]
            frames.append({"frame_id": frame_id, "timestamp_seconds": timestamp,
                           "timestamp_estimated": bool(estimated),
                           "detections": json.loads(result.to_json())})
        if not frames:
            raise ValueError(f"Video contains no readable frames: {args.source}")
    finally:
        capture.release()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"source": str(args.source.resolve()), "fps": fps, "width": width,
               "height": height, "frames": frames}
    (args.output / "video_detections.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved detections for {len(frames)} frames to {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--model", type=Path,
                        default=Path(os.environ.get("YOLOV11_WEIGHTS") or root / ".cache/yolov11/assets/yolo11n.pt"),
                        help="Existing weights; defaults to cached YOLO11n (YOLOV11_WEIGHTS overrides).")
    parser.add_argument("--source", type=Path,
                        default=Path(os.environ.get("YOLOV11_SOURCE") or root / ".cache/yolov11/assets/bus.jpg"),
                        help="Local image/video; defaults to the cached sample (YOLOV11_SOURCE overrides).")
    parser.add_argument("--output", type=Path, default=root / "outputs/yolov11",
                        help="Directory for image results or video_detections.json.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--video", action="store_true", help="Export chronological video_detections.json.")
    parser.add_argument("--max-frames", type=int, help="Optional positive video frame limit.")
    args = parser.parse_args()
    for label, path in (("model", args.model), ("source", args.source)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".cache/yolov11/config"))
    from ultralytics import YOLO

    model = YOLO(str(args.model), task="detect")
    if args.video:
        try:
            detect_video(model, args)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        return
    result = model.predict(
        source=str(args.source), device="cpu", imgsz=args.imgsz,
        conf=args.conf, save=False, verbose=True,
    )[0]
    args.output.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(args.output / "annotated.jpg"))
    (args.output / "detections.json").write_text(result.to_json() + "\n", encoding="utf-8")
    print(f"Saved {len(result.boxes)} detections to {args.output.resolve()}")


if __name__ == "__main__":
    main()
