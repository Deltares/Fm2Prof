"""Entry point for the FM2PROF command-line interface.

For example, to run the CLI, use:
    python -m fm2prof <command> [options]
"""
from fm2prof import cli

if __name__ == "__main__":
    cli.app()
