import copy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest


@pytest.fixture
def sequence():
    frames = []
    for i in range(6):
        detections = [] if i == 3 else [
            {"class": 0, "confidence": 0.15 if i == 1 else 0.9,
             "box": {"x1": 10 + 2 * i, "y1": 10, "x2": 40 + 2 * i, "y2": 70}},
            {"class": 2, "confidence": 0.9,
             "box": {"x1": 100 + i, "y1": 80, "x2": 150 + i, "y2": 120}},
        ]
        frames.append({"frame_id": i, "timestamp_seconds": i / 10, "detections": detections})
    return {"fps": 10.0, "width": 200, "height": 150, "frames": frames}


def test_detection_schema_preserves_empty_frames(repo, tmp_path, sequence):
    module = runpy.run_path(str(repo / "plugins/models/bytetrack/track.py"))
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(sequence))
    assert module["load_detections"](path) == sequence


@pytest.mark.parametrize("case", ["not_object", "fps", "frames", "order", "timestamp", "class", "confidence", "box"])
def test_invalid_detection_schema(repo, tmp_path, sequence, case):
    data = copy.deepcopy(sequence)
    if case == "not_object":
        data = []
    elif case == "fps":
        data["fps"] = 0
    elif case == "frames":
        data["frames"] = []
    elif case == "order":
        data["frames"][1]["frame_id"] = 7
    elif case == "timestamp":
        data["frames"][1]["timestamp_seconds"] = 0
    elif case == "class":
        data["frames"][0]["detections"][0]["class"] = "person"
    elif case == "confidence":
        data["frames"][0]["detections"][0]["confidence"] = 2
    elif case == "box":
        data["frames"][0]["detections"][0]["box"]["x2"] = 5
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(data))
    module = runpy.run_path(str(repo / "plugins/models/bytetrack/track.py"))
    with pytest.raises(ValueError):
        module["load_detections"](path)


def test_missing_input_error_without_ai_dependencies(repo, tmp_path):
    import sys

    result = subprocess.run([sys.executable, str(repo / "plugins/models/bytetrack/track.py"),
                             "--input", str(tmp_path / "missing.json"), "--output", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "No such file" in result.stderr
    assert "Traceback" not in result.stderr


def test_real_bytetrack_identity_and_low_confidence_recovery(repo, tmp_path, sequence):
    python = Path(os.environ.get("YOLOV11_PYTHON", repo / ".cache/yolov11/venv/bin/python"))
    if not python.is_file():
        pytest.skip("Optional YOLO11/ByteTrack environment is not installed")
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(sequence))
    result = subprocess.run([str(python), "-I", str(repo / "plugins/models/bytetrack/track.py"),
                             "--input", str(path), "--output", str(tmp_path)],
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, "YOLO_CONFIG_DIR": str(tmp_path / "config")})
    assert result.returncode == 0, result.stderr
    tracked = json.loads((tmp_path / "tracks.json").read_text())
    assert [f["frame_id"] for f in tracked["frames"]] == list(range(6))
    assert [f["timestamp_seconds"] for f in tracked["frames"]] == [i / 10 for i in range(6)]
    first_ids = {t["class"]: t["track_id"] for t in tracked["frames"][0]["tracks"]}
    assert len(set(first_ids.values())) == 2
    assert tracked["frames"][3]["tracks"] == []
    for i in (1, 2, 4, 5):
        assert {t["class"]: t["track_id"] for t in tracked["frames"][i]["tracks"]} == first_ids


@pytest.mark.parametrize("workflow", ["object_tracking_file", "object_tracking"])
@pytest.mark.parametrize("use_overrides", [False, True])
def test_object_tracking_workflow(repo, tmp_path, workflow, use_overrides):
    import shutil
    import sys
    from collections import Counter

    python = repo / ".cache/yolov11/venv/bin/python"
    weights = repo / ".cache/yolov11/assets/yolo11n.pt"
    video = repo / ".cache/surveillance/short.avi"
    if not all(path.is_file() for path in (python, weights, video)):
        pytest.skip("Optional YOLO11 runtime, weights and short video are not installed")
    shutil.copytree(repo / "plugins", tmp_path / "plugins", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(repo / "projects/surveillance/yolo11_bytetrack",
                    tmp_path / "projects/surveillance/yolo11_bytetrack")
    env = {**os.environ, "YOLO_CONFIG_DIR": str(tmp_path / "config"),
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    for key in ("YOLOV11_PYTHON", "YOLOV11_WEIGHTS", "YOLOV11_SOURCE", "SURVEILLANCE_VIDEO"):
        env.pop(key, None)
    if use_overrides:
        env.update(YOLOV11_PYTHON=str(python), YOLOV11_WEIGHTS=str(weights),
                   YOLOV11_SOURCE=str(video), SURVEILLANCE_VIDEO=str(video))
    else:
        # Reuse the configured installation's external assets/runtime without copying it.
        (tmp_path / ".cache").symlink_to(repo / ".cache", target_is_directory=True)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-m", "eyecore.cli", "--root", str(tmp_path),
                             "run", workflow], env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "outputs/object_tracking"
    tracking_dir = output / ("in_memory" if workflow == "object_tracking" else "tracking")
    tracks = json.loads((tracking_dir / "tracks.json").read_text())
    assert len(tracks["frames"]) == 30
    if workflow == "object_tracking_file":
        detections = json.loads((output / "detections/video_detections.json").read_text())
        assert [(f["frame_id"], f["timestamp_seconds"]) for f in tracks["frames"]] == [
            (f["frame_id"], f["timestamp_seconds"]) for f in detections["frames"]]
    else:
        assert set(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()) == {
            "in_memory/tracks.json", "in_memory/tracks.avi"}
        assert [f["frame_id"] for f in tracks["frames"]] == list(range(30))
        assert all(a["timestamp_seconds"] < b["timestamp_seconds"]
                   for a, b in zip(tracks["frames"], tracks["frames"][1:]))
    counts = Counter(t["track_id"] for f in tracks["frames"] for t in f["tracks"])
    assert max(counts.values()) > 1
    assert (tracking_dir / "tracks.avi").read_bytes().startswith(b"RIFF")


@pytest.mark.parametrize("script", ["plugins/models/yolov11/predict.py",
                                    "projects/surveillance/yolo11_bytetrack/run.py"])
def test_model_environment_override_is_owned_by_module(repo, tmp_path, script):
    import sys

    source = tmp_path / "input.avi"
    source.touch()
    custom_weights = tmp_path / "missing-custom.pt"
    result = subprocess.run([sys.executable, str(repo / script), "--source", str(source),
                             "--output", str(tmp_path / "output")], capture_output=True, text=True,
                            env={**os.environ, "YOLOV11_WEIGHTS": str(custom_weights)})
    assert result.returncode == 2
    assert str(custom_weights) in result.stderr
    assert "model file does not exist" in result.stderr
    assert "Traceback" not in result.stderr


def test_in_memory_deterministic_sequence_and_frame_limit(repo, tmp_path):
    python = repo / ".cache/yolov11/venv/bin/python"
    weights = repo / ".cache/yolov11/assets/yolo11n.pt"
    image = repo / ".cache/yolov11/assets/bus.jpg"
    if not all(path.is_file() for path in (python, weights, image)):
        pytest.skip("Optional YOLO11 environment and assets are not installed")
    video = tmp_path / "stationary.avi"
    code = """import cv2, sys
image = cv2.imread(sys.argv[1])
writer = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*'MJPG'), 10, (810, 1080))
assert writer.isOpened()
for _ in range(6):
    writer.write(image)
writer.release()
"""
    env = {**os.environ, "YOLO_CONFIG_DIR": str(tmp_path / "config"),
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    env.pop("PYTHONPATH", None)
    subprocess.run([str(python), "-I", "-c", code, str(image), str(video)], env=env, check=True)
    output = tmp_path / "result"
    result = subprocess.run([str(python), "-I", str(repo / "projects/surveillance/yolo11_bytetrack/run.py"),
                             "--model", str(weights), "--source", str(video), "--output", str(output),
                             "--max-frames", "4"], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = json.loads((output / "tracks.json").read_text())["frames"]
    assert len(frames) == 4
    initial = {(t["class"], t["track_id"]) for t in frames[0]["tracks"]}
    assert len(initial) >= 2
    for frame in frames[1:]:
        assert {(t["class"], t["track_id"]) for t in frame["tracks"]} == initial
    assert {p.name for p in output.iterdir()} == {"tracks.json", "tracks.avi"}
