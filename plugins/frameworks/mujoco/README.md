# MuJoCo headless simulation

This plugin uses the official `mujoco` Python package, pinned to **3.3.7**.
`simulate.py` loads the small local `pendulum.xml`, initializes its first keyframe
and advances the physics with `mj_step`. It has no EyeCore imports. No renderer,
viewer, display, GPU, external robot model or dataset is needed.

## Install an independent environment

From the EyeCore repository root, using the existing Python 3.11+ core interpreter
only to create a separate environment:

```bash
.venv/bin/python -m venv .cache/mujoco/venv
.cache/mujoco/venv/bin/python -m pip install \
  -r plugins/frameworks/mujoco/requirements.txt
export MUJOCO_PYTHON="$PWD/.cache/mujoco/venv/bin/python"
```

You can instead set `MUJOCO_PYTHON` to an absolute interpreter path in another
environment containing the pinned package. Dependencies are installed only there,
separately from EyeCore's `.venv` and `.cache/yolov11/venv`. The package bundles
the physics engine, so no separate MuJoCo installation is necessary. Its upstream
rendering dependencies are installed with the package but no rendering API is
used by this simulation. `.cache/` and generated `outputs/` are ignored by Git.

## Run independently

```bash
"$MUJOCO_PYTHON" -I plugins/frameworks/mujoco/simulate.py \
  --steps 1000 --output "$PWD/outputs/mujoco-validation/independent.json"
```

`-I` isolates Python from the working directory and Python environment variables.
This command does not require EyeCore installed in the simulation environment.
The XML defaults to the file beside the script, regardless of working directory.

## Run through EyeCore

EyeCore automatically selects `.cache/mujoco/venv/bin/python` on the configured
installation. No export is needed; `MUJOCO_PYTHON` remains an optional override:

```bash
source .venv/bin/activate
eyecore list
eyecore info mujoco
eyecore run mujoco
```

The manifest selects the external interpreter, script and output path; step
count and physics settings remain the simulation script/model's defaults. No core
changes or runtime manager are needed. If your terminal injects other Python
packages through `PYTHONPATH`, prefix commands with `env -u PYTHONPATH` on POSIX.
You can verify headless execution with `DISPLAY` and `WAYLAND_DISPLAY` unset.

The default run advances **1,000 steps**, with a **0.002-second timestep**, for
approximately **2 seconds** of simulation. It writes `outputs/mujoco/result.json`:

```json
{
  "steps": 1000,
  "simulation_time": 2.0000000000000013,
  "joint_positions": [-0.07877350143570805],
  "joint_velocities": [0.8114416303914825]
}
```

These values are from the validated run. Time may differ from exactly 2.0 by
floating-point roundoff, and final state can vary slightly across platforms. The
model has one damped hinge, starts at 0.3 radians, and moves under gravity.
Positions are radians and velocities are radians/second for this hinge model.
The arrays are MuJoCo's `qpos` and `qvel`; their dimensions and units depend on
joint types when using another model.

Override steps, XML or output with explicit arguments:

```bash
eyecore run mujoco -- --steps 250 --output /tmp/mujoco-result.json
"$MUJOCO_PYTHON" plugins/frameworks/mujoco/simulate.py \
  --model /absolute/path/to/model.xml --steps 250 --output /tmp/native-result.json
```

The first keyframe is used if the XML defines one; otherwise the model's default
initial state is used. Steps must be positive. Existing output files are
overwritten. This example exports final state only; it does not render, control
robots, record trajectories or connect simulations to other modules.

## Validation

Validated on Linux with Python 3.11.14 and MuJoCo 3.3.7. Native and EyeCore runs
both exited 0 with display variables unset and produced identical final-state
JSON. Tests also compare both execution paths at 25 steps (0.05 seconds), check
that the joint actually moves and verify finite state values. That integration
test skips if the optional MuJoCo environment is absent; core tests need no
MuJoCo dependency. Run checks with:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

Official references: [Python bindings and installation](https://mujoco.readthedocs.io/en/3.3.7/python.html),
[MJCF XML reference](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html).
