"""A module that can live in any independent repository."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--message", default="Hello from an independent module")
args = parser.parse_args()
print(f"{args.message} (working directory: {Path.cwd()})", flush=True)
