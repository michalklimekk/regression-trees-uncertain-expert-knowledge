"""Run all experiments from the project directory."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def run(script, *extra):
    command = [
        sys.executable,
        str(CODE / script),
        "--results",
        str(RESULTS),
        "--figures",
        str(FIGURES),
        *extra,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=CODE)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    run("simulate_two_leaf.py")
    run("simulate_larger_trees.py")
    run("sensitivity_experiments.py")
    run("real_data_experiments.py")


if __name__ == "__main__":
    main()
