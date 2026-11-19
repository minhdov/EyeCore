import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import Mock

import pytest

from eyecore import EyeCoreError
from eyecore.registry import discover_plugins, load_plugin
from eyecore.runner import prepare_command, run_plugin


def test_independent_execution_and_working_directory(workspace, manifest, capfd):
    path = manifest(code="from pathlib import Path; print(Path.cwd()); Path('result.txt').write_text('done')")
    assert run_plugin(load_plugin(path), workspace) == 0
    assert (path.parent / "result.txt").read_text() == "done"
    assert str(path.parent) in capfd.readouterr().out
    # The module also runs directly, without importing EyeCore.
    result = subprocess.run([sys.executable, "main.py"], cwd=path.parent, capture_output=True)
    assert result.returncode == 0


def test_external_directory_and_explicit_interpreter(workspace, manifest, monkeypatch, tmp_path):
    external = tmp_path / "external repository"
    external.mkdir()
    (external / "main.py").write_text("from pathlib import Path; Path('done').touch()")
    monkeypatch.setenv("MODULE_REPO", str(external))
    monkeypatch.setenv("MODULE_PYTHON", sys.executable)
    path = manifest(runtime={"type": "python", "executable": "${MODULE_PYTHON}"},
                    execution={"working_dir": "${MODULE_REPO}", "command": ["{python}", "main.py"]})
    assert run_plugin(load_plugin(path), workspace) == 0
    assert (external / "done").exists()


def test_arguments_are_literal(workspace, manifest):
    path = manifest(code="import sys,json; from pathlib import Path; Path('args.json').write_text(json.dumps(sys.argv[1:]))")
    injection = "; touch HACKED; $(touch HACKED) `touch HACKED`"
    assert run_plugin(load_plugin(path), workspace, [injection, "value with spaces"]) == 0
    import json
    assert json.loads((path.parent / "args.json").read_text()) == [injection, "value with spaces"]
    assert not (path.parent / "HACKED").exists()


def test_missing_executable(workspace, manifest, capfd):
    path = manifest(runtime={"type": "python", "executable": "/no-such-eyecore-python"})
    assert run_plugin(load_plugin(path), workspace) == 127
    assert "Executable not found" in capfd.readouterr().err


def test_missing_working_directory(workspace, manifest):
    path = manifest(execution={"working_dir": "missing", "command": ["{python}", "main.py"]})
    with pytest.raises(EyeCoreError, match="Working directory does not exist"):
        run_plugin(load_plugin(path), workspace)


def test_failure_exit_code(workspace, manifest):
    assert run_plugin(load_plugin(manifest(code="raise SystemExit(7)")), workspace) == 7


@pytest.mark.parametrize("runtime, command", [
    ("docker", ["docker", "run", "--rm", "example-image"]),
    ("ros2", ["ros2", "launch", "my_robot", "demo.launch.py"]),
    ("native", ["launcher", "--config", "settings.yaml"]),
])
def test_native_launchers_are_not_wrapped(workspace, manifest, runtime, command, monkeypatch):
    path = manifest(runtime={"type": runtime}, execution={"working_dir": ".", "command": command})
    process = Mock()
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr("eyecore.runner.subprocess.Popen", popen)
    assert run_plugin(load_plugin(path), workspace) == 0
    assert popen.call_args.args == (command,)
    assert popen.call_args.kwargs["shell"] is False


def test_template_is_actionable(workspace, manifest):
    plugin = load_plugin(manifest(template=True))
    with pytest.raises(EyeCoreError, match="template: false"):
        run_plugin(plugin, workspace)


@pytest.mark.parametrize("waits", [
    [KeyboardInterrupt, 0],
    [KeyboardInterrupt, subprocess.TimeoutExpired("test", 3), 0],
    [KeyboardInterrupt, subprocess.TimeoutExpired("test", 3), subprocess.TimeoutExpired("test", 3), 0],
])
def test_interruption_cleanup(workspace, manifest, monkeypatch, waits):
    process = Mock(pid=12345)
    process.wait.side_effect = waits
    monkeypatch.setattr("eyecore.runner.subprocess.Popen", Mock(return_value=process))
    send_signal = Mock()
    monkeypatch.setattr("eyecore.runner._signal_process", send_signal)
    assert run_plugin(load_plugin(manifest()), workspace) == 130
    assert send_signal.call_args_list[0].args == (process, signal.SIGINT)
    assert process.wait.call_count == len(waits)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal integration")
def test_real_ctrl_c_stops_child(workspace, manifest):
    path = manifest(code="""from pathlib import Path
import os
import time
Path('ready').write_text(str(os.getpid()))
try:
    time.sleep(60)
except KeyboardInterrupt:
    Path('stopped').touch()
""")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    process = subprocess.Popen(
        [sys.executable, "-m", "eyecore.cli", "--root", str(workspace.root), "run", "test_module"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    import time
    try:
        deadline = time.monotonic() + 10
        while not (path.parent / "ready").exists():
            assert process.poll() is None
            if time.monotonic() >= deadline:
                pytest.fail("Child did not start")
            time.sleep(0.02)
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 130
        assert "Interrupted" in stderr
        assert "Traceback" not in stderr
        assert (path.parent / "stopped").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int((path.parent / "ready").read_text()), 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_add_plugin_without_core_changes(workspace, manifest):
    assert discover_plugins(workspace.plugin_paths) == {}
    manifest(name="new_student_module", folder="students/new", code="print('added')")
    plugins = discover_plugins(workspace.plugin_paths)
    assert run_plugin(plugins["new_student_module"], workspace) == 0


def test_relative_interpreter_path(workspace, manifest):
    path = manifest(runtime={"type": "python", "executable": "./env/bin/python"})
    command, cwd = prepare_command(load_plugin(path), workspace, [])
    assert command[0] == str(path.parent / "env/bin/python")
    assert cwd == path.parent


@pytest.mark.parametrize("use_override", [False, True])
def test_mujoco_independent_and_eyecore_execution(repo, tmp_path, use_override):
    import json
    import math

    python = Path(os.environ.get("MUJOCO_PYTHON", repo / ".cache/mujoco/venv/bin/python"))
    if not python.is_file():
        pytest.skip("Optional MuJoCo environment is not installed")
    env = dict(os.environ)
    env.pop("MUJOCO_PYTHON", None)
    if use_override:
        env["MUJOCO_PYTHON"] = str(python.absolute())
    for key in ("PYTHONPATH", "DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(key, None)
    native_output = tmp_path / "native.json"
    cli_output = tmp_path / "eyecore.json"
    commands = [
        [str(python), "-I", str(repo / "plugins/frameworks/mujoco/simulate.py"),
         "--steps", "25", "--output", str(native_output)],
        [sys.executable, "-m", "eyecore.cli", "--root", str(repo), "run", "mujoco",
         "--", "--steps", "25", "--output", str(cli_output)],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
    native = json.loads(native_output.read_text())
    assert native == json.loads(cli_output.read_text())
    assert native["steps"] == 25
    assert native["simulation_time"] == pytest.approx(0.05)
    assert len(native["joint_positions"]) == len(native["joint_velocities"]) == 1
    assert all(math.isfinite(v) for v in native["joint_positions"] + native["joint_velocities"])
    assert native["joint_positions"][0] != pytest.approx(0.3)
    assert native["joint_velocities"][0] != pytest.approx(0)
