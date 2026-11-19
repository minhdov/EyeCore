from pathlib import Path
import sys

import pytest
import yaml

from eyecore.registry import load_settings


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "projects").mkdir()
    return load_settings(tmp_path)


@pytest.fixture
def manifest(workspace):
    def create(name="test_module", code="print('hello')", folder=None, **updates):
        directory = workspace.root / "plugins" / (folder or str(name))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "main.py").write_text(code)
        data = {
            "name": name, "type": "module",
            "runtime": {"type": "python", "executable": sys.executable},
            "execution": {"working_dir": ".", "command": ["{python}", "main.py"]},
        }
        data.update(updates)
        path = directory / "plugin.yaml"
        path.write_text(yaml.safe_dump(data))
        return path
    return create


@pytest.fixture
def workflow_file(workspace):
    def create(steps, name="pipeline", **updates):
        path = workspace.root / "projects" / name / "workflow.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"name": name, "steps": steps, **updates}))
        return path
    return create


@pytest.fixture
def repo():
    return Path(__file__).resolve().parents[1]
