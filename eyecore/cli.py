"""Command line entry point."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from eyecore import EyeCoreError
from eyecore.registry import discover_plugins, discover_workflows, load_settings
from eyecore.runner import run_plugin, run_workflow

app = typer.Typer(no_args_is_help=True, help="EyeCore — run independent modules and sequential workflows.")


@app.callback()
def main(
    ctx: typer.Context,
    root: Annotated[Path, typer.Option(help="Integration repository root (defaults to current directory).")] = Path("."),
    config: Annotated[Optional[Path], typer.Option(help="YAML configuration; defaults to ROOT/eyecore.yaml.")] = None,
) -> None:
    ctx.obj = (root, config)


def catalog(ctx: typer.Context):
    settings = load_settings(*ctx.obj)
    return settings, discover_plugins(settings.plugin_paths), discover_workflows(settings.project_paths)


@app.command("list")
def list_entries(ctx: typer.Context) -> None:
    """List registered plugins and project workflows, including templates."""
    try:
        _, plugins, workflows = catalog(ctx)
        typer.echo("Plugins:")
        for name, plugin in sorted(plugins.items()):
            typer.echo(f"  {name} ({plugin.type}, {plugin.runtime})" + (" [template]" if plugin.template else ""))
        typer.echo("Workflows:")
        for name in sorted(workflows):
            typer.echo(f"  {name}")
    except EyeCoreError as exc:
        fail(exc)


def select(name, plugins, workflows, kind):
    if kind not in (None, "plugin", "workflow"):
        raise EyeCoreError("--kind must be 'plugin' or 'workflow'.")
    if kind is None and name in plugins and name in workflows:
        raise EyeCoreError(f"Ambiguous name '{name}': both a plugin and workflow. Add --kind plugin or --kind workflow.")
    if kind != "workflow" and name in plugins:
        return "plugin", plugins[name]
    if kind != "plugin" and name in workflows:
        return "workflow", workflows[name]
    raise EyeCoreError(f"Unknown {kind or 'plugin or workflow'} '{name}'. Use 'eyecore list' and check --root/--config.")


@app.command()
def info(
    ctx: typer.Context,
    name: str,
    kind: Annotated[Optional[str], typer.Option(help="Disambiguate: plugin or workflow.")] = None,
) -> None:
    """Show manifest metadata without launching or importing the module."""
    try:
        _, plugins, workflows = catalog(ctx)
        selected, entry = select(name, plugins, workflows, kind)
        typer.echo(f"Name: {entry.name}\nKind: {selected}\nManifest: {entry.manifest}")
        if selected == "plugin":
            typer.echo(f"Type: {entry.type}\nRuntime: {entry.runtime}\nPython: {entry.executable or '(EyeCore interpreter)'}")
            typer.echo(f"Working directory: {entry.working_dir}\nCommand: {entry.command}\nTemplate: {entry.template}")
            if entry.description:
                typer.echo(entry.description)
        else:
            typer.echo(f"Stop on failure: {entry.stop_on_failure}")
            for step in entry.steps:
                typer.echo(f"  {step.plugin}: {step.args}")
    except EyeCoreError as exc:
        fail(exc)


@app.command()
def run(
    ctx: typer.Context,
    name: str,
    args: Annotated[Optional[list[str]], typer.Argument(help="Extra plugin arguments; place after --.")] = None,
    kind: Annotated[Optional[str], typer.Option(help="Disambiguate: plugin or workflow.")] = None,
) -> None:
    """Run a plugin or workflow; preserve process exit codes."""
    try:
        settings, plugins, workflows = catalog(ctx)
        selected, entry = select(name, plugins, workflows, kind)
        if selected == "workflow":
            if args:
                raise EyeCoreError("Workflow arguments belong in each step's args in workflow.yaml.")
            code = run_workflow(entry, plugins, settings)
        else:
            code = run_plugin(entry, settings, args)
    except EyeCoreError as exc:
        fail(exc)
    except KeyboardInterrupt:
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(130)
    raise typer.Exit(code)


def fail(exc: EyeCoreError) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
