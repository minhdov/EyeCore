import shutil
import sys

from typer.testing import CliRunner

from eyecore.cli import app

runner = CliRunner()


def invoke(root, *args):
    return runner.invoke(app, ["--root", str(root), *args])


def test_help(repo):
    for args in [("--help",), ("run", "--help"), ("info", "--help"), ("list", "--help")]:
        result = invoke(repo, *args)
        assert result.exit_code == 0, result.output
        assert "Usage" in result.output


def test_list_and_info(repo):
    result = invoke(repo, "list")
    assert result.exit_code == 0
    for name in ("dummy", "external_echo", "yolov11", "bytetrack", "mujoco", "object_tracking", "object_tracking_file"):
        assert name in result.output
    result = invoke(repo, "info", "yolov11")
    assert result.exit_code == 0
    assert "Template: False" in result.output
    assert "YOLOV11_PYTHON" in result.output
    assert invoke(repo, "info", "object_tracking").exit_code == 0


def test_run_and_extra_arguments(workspace, manifest):
    manifest(code="import sys; raise SystemExit(0 if sys.argv[1:] == ['--value', 'hello world'] else 8)")
    result = invoke(workspace.root, "run", "test_module", "--", "--value", "hello world")
    assert result.exit_code == 0, result.output


def test_exit_code(workspace, manifest):
    manifest(code="raise SystemExit(6)")
    assert invoke(workspace.root, "run", "test_module").exit_code == 6


def test_ambiguity(workspace, manifest, workflow_file):
    manifest(name="shared")
    workflow_file([{"plugin": "shared"}], name="shared")
    for command in ("run", "info"):
        result = invoke(workspace.root, command, "shared")
        assert result.exit_code == 1
        assert "Ambiguous" in result.output
        assert "--kind" in result.output
    assert invoke(workspace.root, "run", "shared", "--kind", "plugin").exit_code == 0
    assert invoke(workspace.root, "run", "shared", "--kind", "workflow").exit_code == 0


def test_actionable_errors(workspace):
    result = invoke(workspace.root, "run", "unknown")
    assert result.exit_code == 1
    assert "eyecore list" in result.output
    assert invoke(workspace.root, "run", "unknown", "--kind", "invalid").exit_code == 1


def test_workflow_rejects_extra_args(workspace, workflow_file):
    workflow_file([{"plugin": "x"}])
    result = invoke(workspace.root, "run", "pipeline", "--", "--input", "x")
    assert result.exit_code == 1
    assert "each step's args" in result.output


def test_template_run_error(workspace, manifest):
    manifest(name="unconfigured", template=True)
    result = invoke(workspace.root, "run", "unconfigured")
    assert result.exit_code == 1
    assert "template" in result.output
    assert "Traceback" not in result.output


def test_mujoco_info(repo):
    result = invoke(repo, "info", "mujoco")
    assert result.exit_code == 0, result.output
    assert "Type: framework" in result.output
    assert "MUJOCO_PYTHON:-" in result.output
    assert "Template: False" in result.output
    assert "simulate.py" in result.output


def test_migrated_external_module(repo, tmp_path, monkeypatch, capfd):
    external = tmp_path / "independent repository"
    external.mkdir()
    shutil.copy(repo / "plugins/modules/external_echo/echo_module.py", external)
    monkeypatch.setenv("EXTERNAL_REPO", str(external))
    monkeypatch.setenv("EXTERNAL_PYTHON", sys.executable)
    assert "external_echo" in invoke(repo, "list").output
    result = invoke(repo, "run", "external_echo", "--", "--message", "External module works")
    assert result.exit_code == 0, result.output
    output = capfd.readouterr().out
    assert "External module works" in output
    assert str(external) in output
