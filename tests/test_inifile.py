from pathlib import Path

import pytest

from fm2prof.IniFile import IniFile

_root_output_dir = None


class Test_IniFile:
    _test_scenarios_output_dirs = [
        ("dummydir", "dummydir"),
        ("dummydir/dummysubdir", "dummydir/dummysubdir"),
        ("../dummysubdir", "../dummysubdir"),
    ]

    def test_when_no_file_path_then_no_exception_is_risen(self):
        # 1. Set up initial test data
        iniFilePath = ""

        # 2. Run test
        IniFile(iniFilePath)
    

    def test_when_non_existent_file_path_then_io_exception_is_risen(self):
        # 1. Set up initial test data
        ini_file_path = "nonexistent_ini_file.ini"

        # 2. Set expectations
        expected_error = "" + "The given file path {} could not be found".format(
            ini_file_path
        )

        # 3. Run test
        with pytest.raises(IOError) as e_info:
            IniFile(ini_file_path)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, (
            ""
            + "Expected exception message {},".format(expected_error)
            + "retrieved {}".format(error_message)
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
