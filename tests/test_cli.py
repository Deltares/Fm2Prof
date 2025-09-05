from pathlib import Path

from typer.testing import CliRunner

from fm2prof.cli import app
from tests.TestUtils import TestUtils

runner = CliRunner()
new_name = "testIni"


def test_version():
    result = runner.invoke(app, ["--version"]   )
    assert result.exit_code == 0


def test_help():
    result = runner.invoke(app, "--help")
    assert result.exit_code == 0


def test_create_project():
    result = runner.invoke(app, f"create {new_name}")
    assert result.exit_code == 0
    assert Path(f"{new_name}.ini").is_file()


def test_check_project():
    result = runner.invoke(app, f"check {new_name}.ini")
    assert result.exit_code == 0

def test_run_project():
    config = TestUtils.get_local_test_file("cases/case_02_compound/fm2prof_config.ini")
    result = runner.invoke(app, ["run", fr"{config}"])
    assert result.exit_code == 0

def test_run_project_with_post_processing():
    config = TestUtils.get_local_test_file("cases/case_02_compound/fm2prof_config.ini")
    result = runner.invoke(app, ["run", fr"{config}", "--post-process"])
    assert result.exit_code == 0
