import os

from fm2prof import Project
from fm2prof.fm2prof_runner import Fm2ProfRunner
from tests.TestUtils import TestUtils


class TestProject:
    def test_when_no_file_path_then_no_exception_is_risen(self):
        # 1. Set up initial test dat
        project = Project()
        assert project is not None

    def test_run_without_input_no_exception_is_raised(self):
        project = Project()
        project.run()

    def test_run_with_inifile(self):
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file("cases/case_02_compound/fm2prof_config.ini")

        # 3. run test
        project = Project(inifile)
        project.run()

        # 4. verify output
        assert project._output_exists()  # noqa: SLF001

    def test_run_with_overwrite_false_output_unchanged(self):
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file("cases/case_02_compound/fm2prof_config.ini")

        # 2. set expectations
        project = Project(inifile)
        time_before = os.path.getmtime(next(project.output_files))

        # 3. run test
        project.run()
        time_after = os.path.getmtime(next(project.output_files))

        # 4. verify output
        assert time_before == time_after

    def test_run_with_overwrite_true_output_has_changed(self):
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file("cases/case_02_compound/fm2prof_config.ini")

        # 2. set expections
        project = Project(inifile)
        time_before = os.path.getmtime(next(project.output_files))

        # 3. run test
        project.run(overwrite=True)
        time_after = os.path.getmtime(next(project.output_files))

        # 4. verify output
        assert time_before != time_after

    def test_if_get_existing_parameter_then_returned(self):
        # 1. Set up initial test dat
        project = None
        value = None

        # 2. Run test
        project = Project()
        value = project.get_parameter("LakeTimeSteps")

        # 3. Verify final expectations
        assert project is not None
        assert value is not None

    def test_if_get_nonexisting_parameter_then_no_exception(self):
        # 1. Set up initial test dat
        project = None
        value = None
        # 2. Run test
        project = Project()
        value = project.get_parameter("IDoNoTExist")

        # 3. Verify final expectations
        assert project is not None
        assert value is None

    def test_if_get_existing_inputfile_then_returned(self):
        # 1. Set up initial test dat
        project = None
        value = None

        # 2. Run test
        project = Project()
        value = project.get_input_file("CrossSectionLocationFile")

        # 3. Verify final expectations
        assert project is not None
        assert value is not None

    def test_if_get_output_directory_then_returned(self):
        # 1. Set up initial test dat
        project = None
        value = None

        # 2. Run test
        project = Project()
        value = project.get_output_directory()

        # 3. Verify final expectations
        assert project is not None
        assert value is not None

    def test_set_parameter(self):
        # 1. Set up initial test dat
        project = None
        value = 150

        # 2. Run test
        project = Project()
        project.set_parameter("LakeTimeSteps", value)

        # 3. Verify final expectations
        assert project.get_parameter("LakeTimeSteps") == value

    def test_set_input_file(self):
        # 1. Set up initial test dat
        project = None
        value = "RandomString"
        # 2. Run test
        project = Project()
        project.set_input_file("CrossSectionLocationFile", value)

        # 3. Verify final expectations
        assert project.get_input_file("CrossSectionLocationFile") == value

    def test_set_output_directory(self, tmp_path):
        # 1. Set up initial test dat
        project = None
        # 2. Run test
        project = Project()
        project.set_output_directory(tmp_path)

    def test_print_configuration(self):
        # 1. Set up initial test dat
        project = None
        value = None
        # 2. Run test
        project = Project()
        value = project.print_configuration()

        # 3. Verify final expectations
        assert value is not None

class TestFm2ProfRunner:
    def test_when_no_file_path_then_no_exception_is_risen(self):
        # 1. Set up initial test dat
        runner = None

        # 2. Run test
        runner = Fm2ProfRunner()

        # 3. Verify final expectations
        assert runner is not None

    def test_given_inifile_then_no_exception_is_risen(self):
        # 1. Set up initial test data
        ini_file_name = "valid_ini_file.ini"
        dir_name = "IniFile"
        test_data_dir = TestUtils.get_local_test_data_dir(dir_name)
        ini_file_path = test_data_dir / ini_file_name
        runner = None

        # 2. Verify the initial expectations
        assert ini_file_path.exists(), f"Test File {ini_file_path} was not found"

        # 3. Run test
        runner = Fm2ProfRunner(ini_file_path)

        # 4. Verify final expectations
        assert runner is not None

