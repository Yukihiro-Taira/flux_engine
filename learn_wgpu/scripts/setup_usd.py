"""Install the official USD backend into an isolated, untracked environment."""

import argparse
from pathlib import Path
import subprocess
import sys
import venv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--runtime-dir",
    type=Path,
    default=Path(__file__).resolve().parent.parent / ".usd-runtime",
)
args = parser.parse_args()
venv.EnvBuilder(with_pip=True).create(args.runtime_dir)
python = args.runtime_dir / (
    "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
)
subprocess.run(
    [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-r",
        str(Path(__file__).with_name("usd-requirements.txt")),
    ],
    check=True,
)
subprocess.run(
    [
        str(python),
        "-c",
        'from pxr import Usd; from PIL import Image; print("OpenUSD ready:", Usd.GetVersion())',
    ],
    check=True,
)
print(f"Runtime: {python}")
