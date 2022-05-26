from pathlib import Path

from typer.testing import CliRunner

from fm2prof.cli import app

runner = CliRunner()
new_name = "testIni"


def test_app():
    result = runner.invoke(app, "--help")
    assert result.exit_code == 0


def test_app_create_new_file():
    result = runner.invoke(app, f"create {new_name}")
    assert result.exit_code == 0
    assert Path(f"{new_name}.ini").is_file()


def test_app_check_project():
    result = runner.invoke(app, f"check {new_name}.ini")
    assert result.exit_code == 0
