"""Run a fixed number of headless MuJoCo steps and export the final state."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("pendulum.xml"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be a positive integer")
    if not args.model.is_file():
        parser.error(f"Model file does not exist: {args.model}")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    result = {
        "steps": args.steps,
        "simulation_time": float(data.time),
        "joint_positions": data.qpos.tolist(),
        "joint_velocities": data.qvel.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Simulated {args.steps} steps ({data.time:.3f} s); saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
