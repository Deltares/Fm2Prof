import datetime
import os
import shutil
import sys
import unittest

import numpy as np
import pytest

from fm2prof import Project
from fm2prof.CrossSection import CrossSection as CS
from fm2prof.Fm2ProfRunner import Fm2ProfRunner
from fm2prof.Import import FmModelData as FMD
from fm2prof.IniFile import IniFile
from tests.TestUtils import TestUtils

_root_output_dir = None


# region // Helpers
def __get_base_output_dir():
    """
    Sets up the necessary data for MainMethodTest
    """
    output_dir = __create_test_root_output_dir("RunWithFiles_Output")
    # Create it if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


def __create_test_root_output_dir(dirName=None):
    """
    Create test output directory so it's easier
    to collect all output afterwards.
    """
    _root_output_dir = os.path.join(os.path.dirname(__file__), "Output")
    if not os.path.exists(_root_output_dir):
        os.mkdir(_root_output_dir)

    if dirName is not None:
        subOutputDir = os.path.join(_root_output_dir, dirName)
        if not os.path.exists(subOutputDir):
            os.mkdir(subOutputDir)
        return subOutputDir

    return _root_output_dir


def __check_and_create_test_case_output_dir(base_output_dir, caseName):
    """
    Helper to split to set up an output directory
    for the generated data of each test case.
    """
    output_directory = base_output_dir + "\\{0}".format(caseName)

    # clean up the test case output directory if it is no empty
    if os.path.exists(output_directory) and os.listdir(output_directory):
        shutil.rmtree(output_directory)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    return output_directory


# endregion


class Test_Project:
    def test_when_no_file_path_then_no_exception_is_risen(self):
        # 1. Set up initial test dat
        project = Project()
        assert project is not None

    def test_run_without_input_no_exception_is_raised(self):
        project = Project()
        project.run()

    def test_if_get_existing_parameter_then_returned(self):
        # 1. Set up initial test dat
        project = None
        value = None

        # 2. Run test
        try:
            project = Project()
            value = project.get_parameter("LakeTimeSteps")
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert project is not None
        assert value is not None

    def test_if_get_nonexisting_parameter_then_no_exception(self):
        # 1. Set up initial test dat
        project = None
        value = None
        # 2. Run test
        try:
            project = Project()
            value = project.get_parameter("IDoNoTExist")
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert project is not None
        assert value is None

    def test_if_get_existing_inputfile_then_returned(self):
        # 1. Set up initial test dat
        project = None
        value = None
        # 2. Run test
        try:
            project = Project()
            value = project.get_input_file("CrossSectionLocationFile")
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert project is not None
        assert value is not None

    def test_if_get_output_directory_then_returned(self):
        # 1. Set up initial test dat
        project = None
        value = None
        # 2. Run test
        try:
            project = Project()
            value = project.get_output_directory()
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert project is not None
        assert value is not None

    def test_set_parameter(self):
        # 1. Set up initial test dat
        project = None
        value = 150
        # 2. Run test
        try:
            project = Project()
            project.set_parameter("LakeTimeSteps", value)
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert project.get_parameter("LakeTimeSteps") == value

    def test_set_input_file(self):
        # 1. Set up initial test dat
        project = None
        value = "RandomString"
        # 2. Run test
        try:
            project = Project()
            project.set_input_file("CrossSectionLocationFile", value)
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert project.get_input_file("CrossSectionLocationFile") == value

    def test_set_output_directory(self):
        # 1. Set up initial test dat
        project = None
        value = "test/subdir"
        # 2. Run test
        try:
            project = Project()
            project.set_output_directory(value)
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations

    def test_print_configuration(self):
        # 1. Set up initial test dat
        project = None
        value = None
        # 2. Run test
        try:
            project = Project()
            value = project.print_configuration()
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert value is not None


class Test_Fm2ProfRunner:
    def test_when_no_file_path_then_no_exception_is_risen(self):
        # 1. Set up initial test dat
        runner = None

        # 2. Run test
        try:
            runner = Fm2ProfRunner()
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 3. Verify final expectations
        assert runner is not None

    def test_given_inifile_then_no_exception_is_risen(self):
        # 1. Set up initial test data
        ini_file_name = "valid_ini_file.ini"
        dir_name = "IniFile"
        test_data_dir = TestUtils.get_local_test_data_dir(dir_name)
        ini_file_path = os.path.join(test_data_dir, ini_file_name)
        runner = None

        # 2. Verify the initial expectations
        assert os.path.exists(ini_file_path), "" "Test File {} was not found".format(
            ini_file_path
        )

        # 3. Run test
        try:
            runner = Fm2ProfRunner(ini_file_path)
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 4. Verify final expectations
        assert runner is not None

    def test_run_with_inifile(self):
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file(
            "cases/case_02_compound/fm2prof_config.ini"
        )

        # 3. run test
        project = Project(inifile)
        project.run()

        # 4. verify output
        assert project._output_exists()

    def test_run_with_overwrite_false_output_unchanged(self):
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file(
            "cases/case_02_compound/fm2prof_config.ini"
        )

        # 2. set expections
        project = Project(inifile)
        time_before = os.path.getmtime(next(project.output_files))

        # 3. run test
        project.run()
        time_after = os.path.getmtime(next(project.output_files))

        # 4. verify output
        assert time_before == time_after

    def test_run_with_overwrite_true_output_has_changed(self):
        # 1. Set up test data
        inifile = TestUtils.get_local_test_file(
            "cases/case_02_compound/fm2prof_config.ini"
        )

        # 2. set expections
        project = Project(inifile)
        time_before = os.path.getmtime(next(project.output_files))

        # 3. run test
        project.run(overwrite=True)
        time_after = os.path.getmtime(next(project.output_files))

        # 4. verify output
        assert time_before != time_after


class ARCHIVED_Test_generate_cross_section_list:
    def test_when_not_given_FmModelData_then_returns_empty_list(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {"DummyKey": "dummyValue"}
        return_value = None

        # 2. Verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            return_value = runner._generate_cross_section_list(input_param_dict, None)
        except Exception as e_info:
            pytest.fail(
                "Exception {}".format(str(e_info))
                + " was given while generating cross sections"
            )

        # 4. Verify final expectations
        assert return_value is not None
        assert len(return_value) == 0

    def test_when_not_given_input_param_dict_then_returns_empty_list(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        fm_model_data_args = (0, 1, 2, 3, {})
        fm_model_data = FMD(fm_model_data_args)
        return_value = None

        # 2. Verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            return_value = runner._generate_cross_section_list(None, fm_model_data)
        except Exception as e_info:
            pytest.fail(
                "Exception {}".format(str(e_info))
                + " was given while generating cross sections"
            )

        # 4. Verify final expectations
        assert return_value is not None
        assert len(return_value) == 0

    def test_when_given_correct_parameters_then_returns_list_with_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {"dummyKey": "dummyValue"}
        test_number_of_css = 2
        test_css_name = "dummy_css"
        css_data_length = 42
        css_data_location = (4, 2)
        css_data_branch_id = 420
        css_data_chainage = 4.2
        id_key = "id"
        length_key = "length"
        xy_key = "xy"
        branchid_key = "branchid"
        chainage_key = "chainage"
        id_keys = []
        length_values = []
        xy_values = []
        branchid_values = []
        chainage_values = []
        for i in range(test_number_of_css):
            valid_mult = i + 1
            id_keys.append(test_css_name + "_" + str(i))
            length_values.append(css_data_length * valid_mult)
            xy_values.append(tuple([valid_mult * x for x in css_data_location]))
            branchid_values.append(css_data_branch_id * valid_mult)
            chainage_values.append(css_data_chainage * valid_mult)
        css_data = {
            id_key: id_keys,
            length_key: length_values,
            xy_key: xy_values,
            branchid_key: branchid_values,
            chainage_key: chainage_values,
        }

        fmd_arg_list = (None, None, None, None, css_data)
        fm_model_data = FMD(fmd_arg_list)
        # 2. Verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            return_css_list = runner._generate_cross_section_list(
                input_param_dict, fm_model_data
            )
        except Exception as e_info:
            pytest.fail(
                "Exception {}".format(str(e_info))
                + " was given while generating cross sections"
            )

        # 4. Verify final expectations
        assert return_css_list is not None
        assert len(return_css_list) == test_number_of_css
        for idx in range(len(return_css_list)):
            valid_mult = idx + 1
            expected_name = test_css_name + "_" + str(idx)
            expected_data_length = css_data_length * valid_mult
            expected_data_location = tuple([valid_mult * x for x in css_data_location])
            expected_data_branch_id = css_data_branch_id * valid_mult
            expected_data_data_chainage = css_data_chainage * valid_mult
            return_css = return_css_list[idx]
            assert (
                return_css.name == expected_name
            ), "" + "Expected name {} but was {}".format(expected_name, return_css.name)
            assert (
                return_css.length == expected_data_length
            ), "" + "Expected length {} but was {}".format(
                expected_data_length, return_css.length
            )
            assert (
                return_css.location == expected_data_location
            ), "" + "Expected location {} but was {}".format(
                expected_data_location, return_css.location
            )
            assert (
                return_css.branch == expected_data_branch_id
            ), "" + "Expected branch {} but was {}".format(
                expected_data_branch_id, return_css.branch
            )
            assert (
                return_css.chainage == expected_data_data_chainage
            ), "" + "Expected chainage {} but was {}".format(
                expected_data_data_chainage, return_css.chainage
            )

    def test_when_given_correct_parameters_then_returns_list_with_only_valid_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        expected_css = 1
        expected_name = "dummy_css"
        expected_data_length = 42
        expected_data_location = (4, 2)
        expected_data_branch_id = 420
        expected_data_data_chainage = 4.2
        css_data = {
            "id": [expected_name],  # , 'dummy_css_1'],
            "length": [expected_data_length],
            "xy": [expected_data_location],
            "branchid": [expected_data_branch_id],
            "chainage": [expected_data_data_chainage],
        }

        fmd_arg_list = (None, None, None, None, css_data)
        fm_model_data = FMD(fmd_arg_list)
        input_param_dict = {"dummyKey": "dummyValue"}
        # 2. Verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            return_css_list = runner._generate_cross_section_list(
                input_param_dict, fm_model_data
            )
        except Exception as e_info:
            pytest.fail(
                "Exception {}".format(str(e_info))
                + " was given while generating cross sections"
            )

        # 4. Verify final expectations
        assert return_css_list is not None
        assert len(return_css_list) == expected_css
        return_css = return_css_list[0]
        assert return_css is not None
        assert (
            return_css.name == expected_name
        ), "" + "Expected name {} but was {}".format(expected_name, return_css.name)
        assert (
            return_css.length == expected_data_length
        ), "" + "Expected length {} but was {}".format(
            expected_data_length, return_css.length
        )
        assert (
            return_css.location == expected_data_location
        ), "" + "Expected location {} but was {}".format(
            expected_data_location, return_css.location
        )
        assert (
            return_css.branch == expected_data_branch_id
        ), "" + "Expected branch {} but was {}".format(
            expected_data_branch_id, return_css.branch
        )
        assert (
            return_css.chainage == expected_data_data_chainage
        ), "" + "Expected chainage {} but was {}".format(
            expected_data_data_chainage, return_css.chainage
        )


class ARCHIVED_Test_generate_cross_section:
    def test_when_no_css_data_is_given_then_expected_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)

        # 2. Set expectations
        expected_error = "No data was given to create a Cross Section"

        # 3. Run test
        with pytest.raises(Exception) as e_info:
            runner._generate_cross_section(
                css_data=None, input_param_dict=None, fm_model_data=None
            )

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, (
            ""
            + "Expected exception message {},".format(expected_error)
            + " retrieved {}".format(error_message)
        )

    def test_when_no_input_param_dict_is_given_then_expected_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = "dummy_css"
        css_data = {"id": test_css_name}
        # 2. Set expectations
        expected_error = (
            "No input parameters (from ini file)"
            + " given for new cross section {}".format(test_css_name)
        )

        # 3. Run test
        with pytest.raises(Exception) as e_info:
            runner._generate_cross_section(
                css_data=css_data, input_param_dict=None, fm_model_data=None
            )

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, (
            ""
            + "Expected exception message {},".format(expected_error)
            + " retrieved {}".format(error_message)
        )

    def test_when_no_fm_model_data_is_given_then_expected_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = "dummy_name"
        css_data = {"id": test_css_name}
        input_param_dict = {"dummyKey": "dummyValue"}

        # 2. Set expectations
        expected_error = "No FM data given for new cross section" + " {}".format(
            test_css_name
        )

        # 3. Run test
        with pytest.raises(Exception) as e_info:
            runner._generate_cross_section(
                css_data=css_data, input_param_dict=input_param_dict, fm_model_data=None
            )

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, (
            ""
            + "Expected exception message {},".format(expected_error)
            + " retrieved {}".format(error_message)
        )

    def test_when_all_parameters_are_correct_then_returns_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = "dummy_css"
        css_data = {"id": test_css_name}
        input_param_dict = {"dummyKey": "dummyValue"}

        css_data_length = 42
        css_data_location = (4, 2)
        css_data_branch_id = 420
        css_data_chainage = 4.2
        css_data = {
            "id": test_css_name,
            "length": css_data_length,
            "xy": css_data_location,
            "branchid": css_data_branch_id,
            "chainage": css_data_chainage,
        }
        css_data_dict = {
            "id": [test_css_name],
            "length": [css_data_length],
            "xy": [css_data_location],
            "branchid": [css_data_branch_id],
            "chainage": [css_data_chainage],
        }
        fmd_arg_list = (None, None, None, None, css_data_dict)
        fm_model_data = FMD(fmd_arg_list)

        # 2. Expectations
        return_css = None

        # 3. Run test
        try:
            return_css = runner._generate_cross_section(
                css_data=css_data,
                input_param_dict=input_param_dict,
                fm_model_data=fm_model_data,
            )
        except Exception as e_info:
            pytest.fail(
                "No expected exception but was risen:" + " {}".format(str(e_info))
            )

        # 4. Verify final expectations
        assert return_css is not None
        assert (
            return_css.name == test_css_name
        ), "" + "Expected name {} but was {}".format(test_css_name, return_css.name)
        assert (
            return_css.length == css_data_length
        ), "" + "Expected length {} but was {}".format(
            css_data_length, return_css.length
        )
        assert (
            return_css.location == css_data_location
        ), "" + "Expected location {} but was {}".format(
            css_data_location, return_css.location
        )
        assert (
            return_css.branch == css_data_branch_id
        ), "" + "Expected branch {} but was {}".format(
            css_data_branch_id, return_css.branch
        )
        assert (
            return_css.chainage == css_data_chainage
        ), "" + "Expected chainage {} but was {}".format(
            css_data_chainage, return_css.chainage
        )


class ARCHIVED_Test_set_fm_data_to_cross_section:
    def test_when_no_cross_section_given_then_no_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        cross_section = None
        input_param_dict = {"dummy_key": "dummy_value"}
        fm_args = (None, None, None, None, None)
        fm_model_data = FMD(fm_args)
        start_time = datetime.datetime.now

        # 2. Set expectations
        assert runner is not None
        assert input_param_dict is not None
        assert fm_model_data is not None
        assert start_time is not None

        # 3. Run test
        try:
            runner._set_fm_data_to_cross_section(
                cross_section=cross_section,
                input_param_dict=input_param_dict,
                fm_model_data=fm_model_data,
                start_time=start_time,
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was thrown: {}".format(str(e_info)))

    def test_when_no_fm_model_data_given_then_no_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {"dummy_key": "dummy_value"}
        cross_section = CS(input_param_dict, "dummy_name", 4.2, (4, 2))
        fm_model_data = None
        start_time = datetime.datetime.now

        # 2. Set expectations
        assert runner is not None
        assert input_param_dict is not None
        assert cross_section is not None
        assert start_time is not None

        # 3. Run test
        try:
            runner._set_fm_data_to_cross_section(
                cross_section=cross_section,
                input_param_dict=input_param_dict,
                fm_model_data=fm_model_data,
                start_time=start_time,
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was thrown: {}".format(str(e_info)))

    def test_when_given_invalid_parameters_then_no_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {"dummy_key": "dummy_value"}
        cross_section = CS(input_param_dict, "dummy_name", 4.2, (4, 2))
        fm_args = (None, None, None, None, None)
        fm_model_data = FMD(fm_args)
        start_time = datetime.datetime.now

        # 2. Set expectations
        assert runner is not None
        assert input_param_dict is not None
        assert cross_section is not None
        assert fm_model_data is not None
        assert start_time is not None

        # 3. Run test
        try:
            runner._set_fm_data_to_cross_section(
                cross_section=cross_section,
                input_param_dict=input_param_dict,
                fm_model_data=fm_model_data,
                start_time=start_time,
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was thrown: {}".format(str(e_info)))

    def test_when_given_correct_values_then_fm_data_set_to_css(self):
        pytest.fail(
            "To do. This test should verify fm_data is set, "
            + "the CS is built, Delta-h corrected, "
            + "reduced number of points, assigned roughness."
        )


class ARCHIVED_Test_get_new_cross_section:
    @pytest.mark.parametrize("css_data", [(None), ({})])
    def test_when_not_given_css_data_then_returns_none(self, css_data):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)

        # 2. Expectations
        return_value = None

        # 3. Run test
        try:
            return_value = runner._get_new_cross_section(
                css_data=css_data, input_param_dict=None
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was risen: {}".format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is None

    def test_when_css_data_id_not_found_then_returns_none(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {"dummyKey": "dummyValue"}
        css_data = {"dummyKey": "dummyValue"}

        # 2. Expectations
        return_value = None

        # 3. Run test
        try:
            return_value = runner._get_new_cross_section(
                css_data=css_data, input_param_dict=input_param_dict
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was risen: {}".format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is None

    def test_when_css_data_misses_rest_of_key_values_then_returns_none(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {"dummyKey": "dummyValue"}
        css_data = {"id": []}

        # 2. Expectations
        return_value = None

        # 3. Run test
        try:
            return_value = runner._get_new_cross_section(
                css_data=css_data, input_param_dict=input_param_dict
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was risen: {}".format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is None

    def test_when_given_css_data_but_no_input_params_then_returns_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = "dummy_css"
        input_param_dict = None
        css_data_length = 42
        css_data_location = (4, 2)
        css_data_branch_id = 420
        css_data_chainage = 4.2
        css_data = {
            "id": test_css_name,
            "length": css_data_length,
            "xy": css_data_location,
            "branchid": css_data_branch_id,
            "chainage": css_data_chainage,
        }

        # 2. Expectations
        return_css = None

        # 3. Run test
        try:
            return_css = runner._get_new_cross_section(
                css_data=css_data, input_param_dict=input_param_dict
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was risen: {}".format(str(e_info)))

        # 4. Verify final expectations
        assert return_css is not None
        assert (
            return_css.name == test_css_name
        ), "" + "Expected name {} but was {}".format(test_css_name, return_css.name)
        assert (
            return_css.length == css_data_length
        ), "" + "Expected length {} but was {}".format(
            css_data_length, return_css.length
        )
        assert (
            return_css.location == css_data_location
        ), "" + "Expected location {} but was {}".format(
            css_data_location, return_css.location
        )
        assert (
            return_css.branch == css_data_branch_id
        ), "" + "Expected branch {} but was {}".format(
            css_data_branch_id, return_css.branch
        )
        assert (
            return_css.chainage == css_data_chainage
        ), "" + "Expected chainage {} but was {}".format(
            css_data_chainage, return_css.chainage
        )

    def test_when_given_valid_arguments_then_returns_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = "dummy_css"
        input_param_dict = {"dummyKey": "dummyValue"}
        css_data_length = 42
        css_data_location = (4, 2)
        css_data_branch_id = 420
        css_data_chainage = 4.2
        css_data = {
            "id": test_css_name,
            "length": css_data_length,
            "xy": css_data_location,
            "branchid": css_data_branch_id,
            "chainage": css_data_chainage,
        }

        # 2. Expectations
        return_css = None

        # 3. Run test
        try:
            return_css = runner._get_new_cross_section(
                css_data=css_data, input_param_dict=input_param_dict
            )
        except Exception as e_info:
            pytest.fail("No expected exception but was risen: {}".format(str(e_info)))

        # 4. Verify final expectations
        assert return_css is not None
        assert (
            return_css.name == test_css_name
        ), "" + "Expected name {} but was {}".format(test_css_name, return_css.name)
        assert (
            return_css.length == css_data_length
        ), "" + "Expected length {} but was {}".format(
            css_data_length, return_css.length
        )
        assert (
            return_css.location == css_data_location
        ), "" + "Expected location {} but was {}".format(
            css_data_location, return_css.location
        )
        assert (
            return_css.branch == css_data_branch_id
        ), "" + "Expected branch {} but was {}".format(
            css_data_branch_id, return_css.branch
        )
        assert (
            return_css.chainage == css_data_chainage
        ), "" + "Expected chainage {} but was {}".format(
            css_data_chainage, return_css.chainage
        )


class ARCHIVED_Test_export_cross_sections:
    @pytest.mark.parametrize("cross_sections", [(None), ([]), ("")])
    def test_when_no_cross_sections_then_does_not_raise(self, cross_sections):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        output_dir = "dummy_dir"

        # 2. run test
        try:
            runner._export_cross_sections(cross_sections, output_dir)
        except Exception as e:
            e_message = str(e)
            pytest.fail("No exception was expected, but given: {}".format(e_message))

    @pytest.mark.parametrize("output_dir", [(None), ([]), ("")])
    def test_when_no_output_dir_then_does_not_raise(self, output_dir):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_params = {}
        css_name = "dummy_name"
        css_length = 0
        css_location = (0, 0)
        test_css = CS(input_params, css_name, css_length, css_location)
        cross_sections = [test_css]

        # 2. run test
        try:
            runner._export_cross_sections(cross_sections, output_dir)
        except Exception as e:
            e_message = str(e)
            pytest.fail("No exception was expected, but given: {}".format(e_message))

    def test_when_given_invalid_parameters_then_does_not_raise(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        output_dir = "dummy_dir"
        input_params = {}
        css_name = "dummy_name"
        css_length = 0
        css_location = (0, 0)
        test_css = CS(input_params, css_name, css_length, css_location)
        cross_sections = [test_css]

        # 2. Set initial expectations
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        assert os.path.exists(output_dir)

        # 3. run test
        try:
            runner._export_cross_sections(cross_sections, output_dir)
        except Exception as e:
            e_message = str(e)
            shutil.rmtree(output_dir)
            pytest.fail("No exception expected but was thrown {}".format(e_message))

        # 4. Clean up directory
        shutil.rmtree(output_dir)

    def test_when_given_valid_parameters_then_css_are_exported(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        output_dir = "dummy_dir"
        input_params = {}
        css_name = "dummy_name"
        css_length = 0
        css_location = (0, 0)
        test_css = CS(input_params, css_name, css_length, css_location)
        cross_sections = [test_css]

        # 2. Set initial expectations
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        assert os.path.exists(output_dir)
        expected_files = [
            "CrossSectionDefinitions.ini",
            "CrossSectionLocations.ini",
            "geometry.csv",
        ]
        # 3. run test
        try:
            runner._export_cross_sections(cross_sections, output_dir)
        except Exception as e:
            e_message = str(e)
            shutil.rmtree(output_dir)
            pytest.fail("No exception expected but was thrown {}".format(e_message))

        # 4. Verify final expectations
        data_in_dir = os.listdir(output_dir)
        assert data_in_dir is not None
        for expected_file in expected_files:
            assert expected_file in data_in_dir

        # 5. Clean up directory
        # shutil.rmtree(output_dir)


class ARCHIVED_Test_calculate_css_correction:
    def test_when_cross_section_not_given_then_no_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {"sdstorage": "1"}
        test_css = None
        starttime = datetime.datetime.now()

        # 2. Set up / verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            runner._perform_2D_volume_correction(test_css)
        except:
            pytest.fail("Unexpected exception while calculating css correction.")

    def test_when_all_parameters_are_correct_then_calculates_css_correction(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)

        css_name = "dummy_name"
        css_length = 0
        css_location = (0, 0)
        test_css = CS(input_param_dict, css_name, css_length, css_location)

        # 2. Set up / verify initial expectations
        assert runner is not None
        assert test_css is not None
        assert not test_css._css_is_corrected

        # 2.1. values required for correction.
        test_css._css_total_volume = np.array([2, 3, 1, 0])
        test_css._fm_total_volume = np.array([2, 3, 1, 0])
        test_css._css_flow_volume = np.array([2, 3, 1, 0])
        test_css._fm_flow_volume = np.array([2, 3, 1, 0])
        test_css._css_z = np.array([0, 1, 2, 3])

        # 3. Run test
        try:
            runner._perform_2D_volume_correction(test_css)
        except:
            pytest.fail("Unexpected exception while calculating css correction.")

        # 4. Verify final expectations.
        assert test_css._css_is_corrected, (
            "" + "The calculation did not set the flag 'is corrected ' to True"
        )


class ARCHIVED_Test_reduce_css_points:
    def test_when_all_parameters_are_correct_then_reduce_points(self):
        # set up test data
        new_number_of_css_points = 25
        old_number_of_css_points = 30

        runner = Fm2ProfRunner(None)
        input_param_dict = {"number_of_css_points": str(new_number_of_css_points)}

        css_name = "dummy_name"
        css_length = 0
        css_location = (0, 0)
        test_css = CS(input_param_dict, css_name, css_length, css_location)

        # initial expectation
        assert runner is not None
        assert test_css is not None
        assert test_css._css_is_reduced is False
        test_css._css_total_width = np.linspace(10, 20, old_number_of_css_points)
        test_css._css_z = np.linspace(0, 10, old_number_of_css_points)
        test_css._css_flow_width = np.linspace(5, 15, old_number_of_css_points)

        # run
        try:
            runner._reduce_css_points(input_param_dict, test_css)
        except:
            pytest.fail("Unexpected exception while reducing css points.")

        # verify the number of poi
        # assert test_css._css_is_reduced is True
        # pytest.fail('Test still needs work.')
