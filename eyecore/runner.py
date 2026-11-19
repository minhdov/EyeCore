"""Run native processes and simple sequential workflows with live terminal logs."""

from __future__ import annotations

import os
import re
from pathlib import Path
import signal
import subprocess
import sys

from eyecore import EyeCoreError
from eyecore.registry import Plugin, Settings, Workflow


def prepare_command(plugin: Plugin, settings: Settings, args: list[str]) -> tuple[list[str], Path]:
    if plugin.template:
        raise EyeCoreError(
            f"'{plugin.name}' is a configuration template ({plugin.manifest}). "
            "Install its external runtime/assets, configure its paths and command, "
            "then set template: false. See README.md."
        )
    tokens = {"root": str(settings.root), "plugin_dir": str(plugin.manifest.parent)}
    python = settings.render(plugin.executable or sys.executable, **tokens)
    # Relative interpreter paths, like working directories, are manifest-relative.
    python_path = Path(python).expanduser()
    if not python_path.is_absolute() and ("/" in python or "\\" in python):
        python_path = plugin.manifest.parent / python_path
    tokens["python"] = str(python_path)
    cwd = Path(settings.render(plugin.working_dir, **tokens)).expanduser()
    if not cwd.is_absolute():
        cwd = plugin.manifest.parent / cwd
    cwd = cwd.resolve()
    if not cwd.is_dir():
        raise EyeCoreError(f"Working directory does not exist: {cwd}. Check {plugin.manifest}.")
    command = [settings.render(arg, **tokens) for arg in plugin.command + args]
    if not command[0].strip():
        raise EyeCoreError(f"{plugin.manifest}: executable resolves to an empty string.")
    return command, cwd


def _signal_process(process: subprocess.Popen, sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGINT:
            process.terminate()
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def run_plugin(plugin: Plugin, settings: Settings, args: list[str] | None = None) -> int:
    command, cwd = prepare_command(plugin, settings, args or [])
    try:
        # Streams are inherited: logs are live and are never buffered by EyeCore.
        process = subprocess.Popen(command, cwd=cwd, shell=False, start_new_session=os.name == "posix")
    except FileNotFoundError:
        print(f"Executable not found: {command[0]}. Install it or fix runtime.executable/command.", file=sys.stderr)
        return 127
    except OSError as exc:
        print(f"Cannot launch {command[0]} in {cwd}: {exc}", file=sys.stderr)
        return 126
    try:
        code = process.wait()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the active process.", file=sys.stderr)
        _signal_process(process, signal.SIGINT)
        try:
            process.wait(timeout=3)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            _signal_process(process, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                _signal_process(process, signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
                process.kill()
                process.wait()
        return 130
    return code if code >= 0 else 128 - code


def run_workflow(workflow: Workflow, plugins: dict[str, Plugin], settings: Settings) -> int:
    # Validate all references before starting any process.
    for step in workflow.steps:
        if step.plugin not in plugins:
            raise EyeCoreError(f"{workflow.manifest}: unknown plugin '{step.plugin}'. Use 'eyecore list'.")
    failure = 0
    for i, step in enumerate(workflow.steps, 1):
        print(f"[{i}/{len(workflow.steps)}] {step.plugin}", flush=True)
        args = [re.sub(r"(?<!\$)\{workflow_dir\}", lambda _: str(workflow.manifest.parent), arg)
                for arg in step.args]
        code = run_plugin(plugins[step.plugin], settings, args)
        if code:
            failure = failure or code
            if workflow.stop_on_failure or code == 130:
                return code
    return failure
