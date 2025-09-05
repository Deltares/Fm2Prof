"""CLI for Fm2Prof."""

from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from fm2prof import Project, __version__
from fm2prof.ini_file import IniFile
from fm2prof.utils import Compare1D2D, VisualiseOutput

app = typer.Typer()


def _display_version(value: bool) -> None:  # noqa:FBT001
    if value:
        typer.echo(f"Fm2Prof v{__version__}")
        raise typer.Exit


@app.command("create")
def cli_create_new_project(projectname: str) -> None:
    """Create a new project configuration from scratch, then exit."""
    inifile = IniFile().print_configuration()
    ini_path = Path(f"{projectname}.ini")

    with ini_path.open("w") as f:
        f.write(inifile)
    typer.echo(f"{ini_path} written to file")
    raise typer.Exit


@app.command("check")
def cli_check_project(projectname: str) -> None:
    """Load project, check filepaths, print errors then exit."""
    cf = Path(projectname).with_suffix(".ini")
    Project(cf)
    raise typer.Exit


@app.command("compare")
def cli_compare_1d2d(
    projectname: str, output_1d: str, output_2d: str, routes: str
) -> None:
    """BETA FUNCTIONALITY - compares 1D and 2D results."""
    cf = Path(projectname).with_suffix(".ini")
    project = Project(cf)

    path_1d = Path(output_1d)
    path_2d = Path(output_2d)

    plotter = Compare1D2D(
        project=project, path_1d=path_1d, path_2d=path_2d, routes=routes
    )

    plotter.eval()


@app.command("run")
def cli_load_project(
    projectname: str,
    *,
    overwrite: bool = typer.Option(
        False,  # noqa: FBT003
        "--overwrite",
        "-o",
        help="Overwrite if output already exists",
    ),
    pp: bool = typer.Option(
        False,  # noqa: FBT003
        "--post-process",
        "-p",
        help="Post-process the results, generates figures",
    ),
    ) -> None:
    """Load and run a project."""
    cf = Path(projectname).with_suffix(".ini")
    project = Project(cf)
    project.run(overwrite=overwrite)

    if pp:
        vis = VisualiseOutput(
            project.get_output_directory(), logger=project.get_logger(),
        )
        for css in tqdm(vis.cross_sections):
            vis.figure_cross_section(css)

    raise typer.Exit


@app.callback()
def cli(
    version: Optional[bool] = typer.Option(  # noqa: ARG001, UP007
        None,
        "--version",
        "-v",
        help="Show the application version and exit",
        callback=_display_version,
        is_eager=True,
    ),
) -> None:
    """Fm2Prof Command-line interface."""
    typer.echo("Welcome to Fm2Prof")
