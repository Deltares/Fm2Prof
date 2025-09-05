import re
import typing
from pathlib import Path
import logging
import pytest

from fm2prof.ini_file import IniFile

_root_output_dir = None



class TestIniFile:
    _test_scenarios_output_dirs: typing.ClassVar = [
        ("dummydir", "dummydir"),
        ("dummydir/dummysubdir", "dummydir/dummysubdir"),
        ("../dummysubdir", "../dummysubdir"),
    ]

    def test_initialise_without_parameters(self):
        # 1. Set up initial test data

        # 2. Run test
        IniFile()

    def test_when_non_existent_file_path_then_io_exception_is_risen(self):
        # 1. Set up initial test data
        ini_file_path = "nonexistent_ini_file.ini"

        # 2. Set expectations
        expected_error = "" + f"The given file path {ini_file_path} could not be found"

        # 3. Run test
        with pytest.raises(FileNotFoundError) as e_info:
            IniFile(ini_file_path)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, (
            "" + f"Expected exception message {expected_error}," + f"retrieved {error_message}"
        )

    @pytest.mark.parametrize("output_dir, expected_value", _test_scenarios_output_dirs)
    def test_set_output_dir_with_valid_input(self, output_dir, expected_value):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        new_output_dir = None

        # 2. Run test
        new_output_dir = iniFile.set_output_directory(output_dir)

        # 3. Verify final expectations
        assert Path(expected_value) == new_output_dir.relative_to(Path().cwd())


    def test_print_configuration(self, tmp_path: Path):
        # 1. Set up initial test data
        ini_file = IniFile()
        output_file = tmp_path / "default_ini_file.ini"

        # 2. Run test
        ini_str = ini_file.print_configuration()

        with output_file.open("w") as f:
            f.write(ini_str)

        # 3. Verify final expectations
        assert output_file.exists()

    def test_missing_input_value_logs_error(self, tmp_path: Path, mocker):
        # 1. Set up initial test data
        ini_file = IniFile(logger=IniFile.create_logger())
        mocked_logger = mocker.patch.object(ini_file, "set_logger_message")
        output_file = tmp_path / "default_ini_file.ini"

        # 2. Run test
        ini_str = ini_file.print_configuration()

        with output_file.open("w") as f:
            f.write(ini_str)

        # 3. Verify final expectations
        ini_file.load_configuration_from_file(output_file)

        mocked_logger.assert_any_call("Could not find input file: CrossSectionLocationFile", "error")

    def test_missing_parameter_logs_warning(self, tmp_path: Path, mocker):
        # 1. Set up initial test data
        ini_file = IniFile(logger=IniFile.create_logger())
        mocked_logger = mocker.patch.object(ini_file, "set_logger_message")
        output_file = tmp_path / "default_ini_file.ini"

        # 2. Run test
        ini_str = ini_file.print_configuration()

        # remove a key from the ini string
        ini_str = re.sub(r".*defaultsection.*\n?", "", ini_str, flags=re.IGNORECASE)

        with output_file.open("w") as f:
            f.write(ini_str)

        # 3. Verify final expectations
        ini_file.load_configuration_from_file(output_file)

        mocked_logger.assert_any_call("Missing key in configuration [parameters]: defaultsection. Using default value",
        "warning")
