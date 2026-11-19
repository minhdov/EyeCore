"""Manifest discovery and portable path configuration. No framework imports."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eyecore import EyeCoreError


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EyeCoreError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EyeCoreError(f"{path}: expected a YAML mapping.")
    return data


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EyeCoreError(f"{label}: expected a nonempty string.")
    if "\0" in value:
        raise EyeCoreError(f"{label}: null characters are not allowed.")
    return value


def string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise EyeCoreError(f"{label}: expected a list of strings.")
    if any("\0" in v for v in value):
        raise EyeCoreError(f"{label}: null characters are not allowed.")
    if nonempty and (not value or not value[0].strip()):
        raise EyeCoreError(f"{label}: command must contain an executable.")
    return value


def checked_name(value: Any, label: str) -> str:
    name = require_text(value, label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise EyeCoreError(f"{label}: use letters, numbers, underscores, dots or hyphens.")
    return name


@dataclass
class Settings:
    root: Path
    plugin_paths: list[Path]
    project_paths: list[Path]
    variables: dict[str, str] = field(default_factory=dict)

    def expand(self, value: str) -> str:
        # Substitution is deliberately single-pass: no evaluation, shell or recursion.
        variables = {**self.variables, **os.environ}

        def replace(match: re.Match) -> str:
            key = match.group(1)
            fallback = match.group(2)
            if fallback is not None and not variables.get(key):
                return fallback
            if key not in variables:
                raise EyeCoreError(
                    f"Variable {key} is not set. Export it or add it to config 'variables'."
                )
            return variables[key]

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", replace, value)

    def render(self, value: str, **tokens: str) -> str:
        # Replace only known tokens; literal braces in command arguments stay literal.
        if tokens:
            value = re.sub(r"(?<!\$)\{(" + "|".join(tokens) + r")\}",
                           lambda match: tokens[match.group(1)], value)
        return self.expand(value)


def load_settings(root: Path, config: Path | None = None) -> Settings:
    root = root.expanduser().resolve()
    config_path = config.expanduser().resolve() if config else root / "eyecore.yaml"
    data = read_yaml(config_path) if config or config_path.is_file() else {}
    if any(not isinstance(key, str) for key in data):
        raise EyeCoreError(f"{config_path}: setting names must be strings.")
    unknown = set(data) - {"variables", "plugin_paths", "project_paths"}
    if unknown:
        raise EyeCoreError(f"{config_path}: unknown settings: {', '.join(sorted(unknown))}")
    variables = data.get("variables", {})
    if not isinstance(variables, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in variables.items()
    ):
        raise EyeCoreError(f"{config_path}: variables must map names to strings.")
    settings = Settings(root, [], [], variables)
    for key, default, env in (
        ("plugin_paths", "plugins", "EYECORE_PLUGIN_PATHS"),
        ("project_paths", "projects", "EYECORE_PROJECT_PATHS"),
    ):
        paths = [root / default]
        for raw in string_list(data.get(key, []), f"{config_path}: {key}"):
            path = Path(settings.expand(raw)).expanduser()
            paths.append(path if path.is_absolute() else config_path.parent / path)
        for raw in filter(None, os.environ.get(env, "").split(os.pathsep)):
            path = Path(settings.expand(raw)).expanduser()
            paths.append(path if path.is_absolute() else root / path)
        setattr(settings, key, [p.resolve() for p in paths])
    return settings


def discover_files(paths: list[Path], filename: str) -> list[Path]:
    found = set()
    for path in paths:
        if not path.exists():
            raise EyeCoreError(f"Discovery path does not exist: {path}. Create it or fix the configuration.")
        if not path.is_dir():
            raise EyeCoreError(f"Discovery path must be a directory: {path}")
        found.update(p.resolve() for p in path.rglob(filename) if p.is_file())
    return sorted(found)


@dataclass
class Plugin:
    name: str
    type: str
    runtime: str
    executable: str | None
    working_dir: str
    command: list[str]
    manifest: Path
    description: str = ""
    template: bool = False


def load_plugin(path: Path) -> Plugin:
    data = read_yaml(path)
    prefix = str(path)
    name = checked_name(data.get("name"), f"{prefix}: name")
    kind = require_text(data.get("type"), f"{prefix}: type")
    if kind not in {"model", "framework", "dataset", "robot", "module"}:
        raise EyeCoreError(f"{prefix}: type must be model, framework, dataset, robot or module.")
    runtime, execution = data.get("runtime"), data.get("execution")
    if not isinstance(runtime, dict) or not isinstance(execution, dict):
        raise EyeCoreError(f"{prefix}: runtime and execution must be mappings.")
    runtime_type = runtime.get("type")
    if runtime_type not in ("python", "native", "docker", "ros2"):
        raise EyeCoreError(f"{prefix}: runtime.type must be python, native, docker or ros2.")
    executable = runtime.get("executable")
    if executable is not None:
        require_text(executable, f"{prefix}: runtime.executable")
    if executable and runtime_type != "python":
        raise EyeCoreError(f"{prefix}: runtime.executable is only used by python runtimes; put native launchers in command.")
    command = string_list(execution.get("command"), f"{prefix}: execution.command", nonempty=True)
    if any("{python}" in arg for arg in command) and runtime_type != "python":
        raise EyeCoreError(f"{prefix}: {{python}} requires runtime.type: python.")
    working_dir = require_text(execution.get("working_dir"), f"{prefix}: execution.working_dir")
    description = data.get("description", "")
    if not isinstance(description, str) or not isinstance(data.get("template", False), bool):
        raise EyeCoreError(f"{prefix}: description must be a string and template must be a boolean.")
    return Plugin(name, kind, runtime_type, executable, working_dir, command,
                  path.resolve(), description, data.get("template", False))


def discover_plugins(paths: list[Path]) -> dict[str, Plugin]:
    plugins: dict[str, Plugin] = {}
    for path in discover_files(paths, "plugin.yaml"):
        plugin = load_plugin(path)
        if plugin.name in plugins:
            raise EyeCoreError(f"Duplicate plugin '{plugin.name}': {plugins[plugin.name].manifest} and {path}")
        plugins[plugin.name] = plugin
    return plugins


@dataclass
class Step:
    plugin: str
    args: list[str]


@dataclass
class Workflow:
    name: str
    steps: list[Step]
    manifest: Path
    stop_on_failure: bool = True


def load_workflow(path: Path) -> Workflow:
    data = read_yaml(path)
    name = checked_name(data.get("name"), f"{path}: name")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise EyeCoreError(f"{path}: steps must be a nonempty list.")
    steps = []
    for i, step in enumerate(raw_steps, 1):
        if not isinstance(step, dict):
            raise EyeCoreError(f"{path}: step {i} must be a mapping.")
        plugin = require_text(step.get("plugin"), f"{path}: step {i} plugin")
        args = string_list(step.get("args", []), f"{path}: step {i} args")
        steps.append(Step(plugin, args))
    stop = data.get("stop_on_failure", True)
    if not isinstance(stop, bool):
        raise EyeCoreError(f"{path}: stop_on_failure must be a boolean.")
    return Workflow(name, steps, path.resolve(), stop)


def discover_workflows(paths: list[Path]) -> dict[str, Workflow]:
    workflows: dict[str, Workflow] = {}
    for path in discover_files(paths, "workflow.yaml"):
        workflow = load_workflow(path)
        if workflow.name in workflows:
            raise EyeCoreError(f"Duplicate workflow '{workflow.name}': {workflows[workflow.name].manifest} and {path}")
        workflows[workflow.name] = workflow
    return workflows
