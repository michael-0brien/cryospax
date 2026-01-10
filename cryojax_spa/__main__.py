import os
import sys
from pathlib import Path


def main():
    help_message = (
        "Welcome to the `cryojax-spa` command line interface. "
        "Invoke a program with syntax: 'cryojax-spa example-program ...'"
    )
    if len(sys.argv) < 2:
        print(help_message)
        sys.exit(1)
    subcommand = sys.argv[1]
    args = sys.argv[2:]
    script_directory = Path(__file__).parent / "programs"
    path_to_program = script_directory / f"{subcommand}.py"
    if path_to_program.exists():
        command = [str(path_to_program)] + args
        os.execv(sys.executable, [sys.executable, *command])
    else:
        print(f"Error: could not find program 'cryojax-spa {subcommand}'.")
        sys.exit(1)
