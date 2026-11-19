import sys

import pytest
import yaml

from eyecore import EyeCoreError
from eyecore.registry import discover_plugins, load_plugin, load_settings, read_yaml


def test_nested_discovery_and_overlapping_roots(workspace, manifest):
    path = manifest(folder="models/student/new_module")
    plugins = discover_plugins([*workspace.plugin_paths, path.parent])
    assert list(plugins) == ["test_module"]
    plugin = plugins["test_module"]
    assert (plugin.type, plugin.runtime, plugin.working_dir) == ("module", "python", ".")
    assert plugin.executable == sys.executable


def test_duplicate_names(workspace, manifest):
    manifest(folder="one")
    manifest(folder="two")
    with pytest.raises(EyeCoreError, match="Duplicate plugin"):
        discover_plugins(workspace.plugin_paths)


@pytest.mark.parametrize("updates", [
    {"name": "bad name"}, {"name": 123}, {"type": "unknown"}, {"type": []},
    {"runtime": []}, {"runtime": {"type": "conda"}},
    {"runtime": {"type": []}},
    {"runtime": {"type": "python", "executable": 123}},
    {"runtime": {"type": "native", "executable": "python"}},
    {"runtime": {"type": "native"}},
    {"execution": {}}, {"execution": "oops"},
    {"execution": {"working_dir": ".", "command": "python main.py"}},
    {"execution": {"working_dir": ".", "command": []}},
    {"execution": {"working_dir": ".", "command": [123]}},
    {"execution": {"working_dir": ".", "command": [""]}},
    {"execution": {"working_dir": "", "command": ["echo"]}},
    {"description": 3}, {"template": "true"},
    {"execution": {"working_dir": ".", "command": ["bad\0command"]}},
    {"execution": {"working_dir": "bad\0path", "command": ["python"]}},
])
def test_invalid_manifest(manifest, updates):
    with pytest.raises(EyeCoreError):
        load_plugin(manifest(**updates))


@pytest.mark.parametrize("text", ["[", "[]", "null", "!!python/object:os.system {}"])
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / "plugin.yaml"
    path.write_text(text)
    with pytest.raises(EyeCoreError):
        read_yaml(path)


def test_missing_manifest(tmp_path):
    with pytest.raises(EyeCoreError, match="Cannot read"):
        load_plugin(tmp_path / "missing.yaml")


def test_config_and_environment_paths(workspace, tmp_path, monkeypatch):
    external = tmp_path / "separate repo"
    external.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "settings.yaml"
    config.write_text(yaml.safe_dump({
        "plugin_paths": ["../separate repo"],
        "project_paths": ["${PROJECTS}"],
        "variables": {"PROJECTS": str(workspace.root / "projects"), "VALUE": "config"},
    }))
    monkeypatch.setenv("EYECORE_PLUGIN_PATHS", str(external))
    monkeypatch.setenv("VALUE", "environment")
    settings = load_settings(workspace.root, config)
    assert external in settings.plugin_paths
    assert settings.expand("${VALUE}") == "environment"
    assert settings.project_paths[-1] == workspace.root / "projects"
    with pytest.raises(EyeCoreError, match="NOT_SET"):
        settings.expand("${NOT_SET}")


@pytest.mark.parametrize("data", [{"variables": []}, {"variables": {"X": 1}},
                                  {"plugin_paths": "foo"}, {"unknown": "x"}, {1: "x"}])
def test_invalid_config(workspace, data):
    config = workspace.root / "eyecore.yaml"
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(EyeCoreError):
        load_settings(workspace.root)


def test_missing_discovery_path(tmp_path):
    with pytest.raises(EyeCoreError, match="Discovery path does not exist"):
        discover_plugins([tmp_path / "missing"])


def test_template_discovery_does_not_require_environment(workspace, manifest, monkeypatch):
    monkeypatch.delenv("UNCONFIGURED_PYTHON", raising=False)
    manifest(name="unconfigured", template=True,
             runtime={"type": "python", "executable": "${UNCONFIGURED_PYTHON}"})
    plugins = discover_plugins(workspace.plugin_paths)
    assert plugins["unconfigured"].template


@pytest.mark.parametrize("value", [None, "", "/custom environment/bin/python"])
def test_optional_environment_fallback(workspace, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OPTIONAL_PYTHON", raising=False)
    else:
        monkeypatch.setenv("OPTIONAL_PYTHON", value)
    expected = value or str(workspace.root / ".cache/runtime/bin/python")
    assert workspace.render("${OPTIONAL_PYTHON:-{root}/.cache/runtime/bin/python}",
                            root=str(workspace.root)) == expected


def test_fallback_substitution_is_single_pass(workspace, monkeypatch):
    monkeypatch.delenv("OPTIONAL_PYTHON", raising=False)
    assert workspace.expand("${OPTIONAL_PYTHON:-literal}") == "literal"
    monkeypatch.setenv("OPTIONAL_PYTHON", "${DO_NOT_EXPAND}")
    assert workspace.expand("${OPTIONAL_PYTHON:-literal}") == "${DO_NOT_EXPAND}"
