# EyeCore

**A Lightweight, Modular AI Integration Framework**

EyeCore simplifies the integration and execution of independent AI modules, external frameworks, and research projects across Computer Vision, Surveillance, Robotics, Embodied AI, Industrial AI, and beyond.

## Features

- **Modular Plugins** — Integrate AI modules and external frameworks independently.
- **Flexible Runtimes** — Use Python environments, Docker, or native executables.
- **Reusable Projects** — Combine multiple plugins into application workflows.
- **Simple CLI** — Run modules and projects through a unified interface.
- **Extensible** — Add new plugins without modifying the Core.

## Installation

```bash
git clone <repository-url>
cd EyeCore
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```bash
# List available plugins and projects
eyecore list

# Inspect a plugin
eyecore info yolov11

# Run an AI module
eyecore run yolov11

# Run an external framework
eyecore run mujoco

# Run a project
eyecore run object_tracking
```

Plugins may require additional dependencies, model weights, or runtime setup. Refer to their respective documentation.

## Project Structure

```text
EyeCore/
├── eyecore/       # Core CLI, Registry, Runner
├── plugins/       # AI modules and framework integrations
├── projects/      # Application workflows
├── tests/         # Regression tests
└── pyproject.toml
```

## Adding Plugins

Create a plugin manifest under `plugins/` specifying the runtime and execution command. Plugins can use independently developed repositories and environments.

Refer to existing plugin implementations for examples.

## License

See the repository license, if provided. Third-party modules and frameworks retain their respective licenses.