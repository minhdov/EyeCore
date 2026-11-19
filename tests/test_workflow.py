import json
import shutil

import pytest
import yaml

from eyecore import EyeCoreError
from eyecore.registry import discover_plugins, discover_workflows, load_workflow
from eyecore.runner import run_workflow


def test_file_based_workflow(workspace, manifest, workflow_file):
    manifest(name="producer", code="""import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'value': 7}))
""")
    manifest(name="consumer", code="""import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps({'result': data['value'] * 2}))
""")
    workflow_file([
        {"plugin": "producer", "args": ["{root}/input.json"]},
        {"plugin": "consumer", "args": ["{root}/input.json", "{root}/result.json"]},
    ])
    plugins = discover_plugins(workspace.plugin_paths)
    workflows = discover_workflows(workspace.project_paths)
    assert run_workflow(workflows["pipeline"], plugins, workspace) == 0
    assert json.loads((workspace.root / "result.json").read_text()) == {"result": 14}


@pytest.mark.parametrize("stop", [True, False])
def test_failure_propagation(workspace, manifest, workflow_file, stop):
    manifest(name="fail", code="raise SystemExit(9)")
    success = manifest(name="success", code="from pathlib import Path; Path('ran').touch()")
    path = workflow_file([{"plugin": "fail"}, {"plugin": "success"}], stop_on_failure=stop)
    assert run_workflow(load_workflow(path), discover_plugins(workspace.plugin_paths), workspace) == 9
    assert (success.parent / "ran").exists() == (not stop)


def test_unknown_plugin_preflight(workspace, manifest, workflow_file):
    path = manifest(code="from pathlib import Path; Path('ran').touch()")
    workflow = load_workflow(workflow_file([{"plugin": "test_module"}, {"plugin": "missing"}]))
    with pytest.raises(EyeCoreError, match="unknown plugin 'missing'"):
        run_workflow(workflow, discover_plugins(workspace.plugin_paths), workspace)
    assert not (path.parent / "ran").exists()


def test_variables_and_tokens_remain_distinct(workspace, manifest, workflow_file, monkeypatch):
    monkeypatch.setenv("python", "configured python value")
    monkeypatch.setenv("workflow_dir", "configured workflow value")
    monkeypatch.setenv("LITERAL", "${DO_NOT_EXPAND_AGAIN}")
    plugin = manifest(code="import sys,json; from pathlib import Path; Path('args.json').write_text(json.dumps(sys.argv[1:]))")
    workflow = workflow_file([{"plugin": "test_module", "args": [
        "${python}", "${workflow_dir}", "{workflow_dir}", "${LITERAL}",
    ]}])
    assert run_workflow(load_workflow(workflow), discover_plugins(workspace.plugin_paths), workspace) == 0
    assert json.loads((plugin.parent / "args.json").read_text()) == [
        "configured python value", "configured workflow value", str(workflow.parent),
        "${DO_NOT_EXPAND_AGAIN}",
    ]


@pytest.mark.parametrize("data", [
    {"name": "demo", "steps": []}, {"name": "demo", "steps": "plugin"},
    {"name": "demo", "steps": ["plugin"]},
    {"name": "demo", "steps": [{"args": []}]},
    {"name": "demo", "steps": [{"plugin": "x", "args": "string"}]},
    {"name": "demo", "steps": [{"plugin": "x"}], "stop_on_failure": "false"},
])
def test_invalid_workflows(tmp_path, data):
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(EyeCoreError):
        load_workflow(path)


def test_duplicate_workflow_names(workspace, workflow_file):
    path = workflow_file([{"plugin": "x"}])
    other = workspace.root / "projects" / "nested"
    other.mkdir()
    shutil.copy(path, other / "workflow.yaml")
    with pytest.raises(EyeCoreError, match="Duplicate workflow"):
        discover_workflows(workspace.project_paths)


def test_interruption_always_stops_workflow(workspace, manifest, workflow_file, monkeypatch):
    manifest()
    path = workflow_file([{"plugin": "test_module"}, {"plugin": "test_module"}], stop_on_failure=False)
    from unittest.mock import Mock
    run = Mock(return_value=130)
    monkeypatch.setattr("eyecore.runner.run_plugin", run)
    assert run_workflow(load_workflow(path), discover_plugins(workspace.plugin_paths), workspace) == 130
    assert run.call_count == 1
