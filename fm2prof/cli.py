from pathlib import Path
from typing import Optional

import typer

from fm2prof import Project, __version__
from fm2prof.IniFile import IniFile

app = typer.Typer()


def _display_version(value: bool) -> None:
    if value:
        typer.echo(f"Fm2Prof v{__version__}")
        raise typer.Exit()


@app.command("create")
def cli_create_new_project(projectname: str):
    """Creates a new project configuration from scratch, then exit"""
    inifile = IniFile().print_configuration()
    ini_path = f"{projectname}.ini"

    with open(ini_path, "w") as f:
        f.write(inifile)
    typer.echo(f"{ini_path} written to file")
    raise typer.Exit()


@app.command("check")
def cli_check_project(configuration_file: str) -> None:
    """Load project, check filepaths, print errors then exit"""
    cf = Path(configuration_file).with_suffix(".ini")
    project = Project(cf)
    raise typer.Exit()


@app.command("run")
def cli_load_project(configuration_file: str) -> None:
    """Loads and runs a project"""
    project = Project(configuration_file).with_suffix(".ini")
    project.run()
    raise typer.Exit()


@app.callback()
def cli(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application version and exit",
        callback=_display_version,
        is_eager=True,
    )
) -> None:
    typer.echo("Welcome to Fm2Prof")
    return


def main():
    app()
