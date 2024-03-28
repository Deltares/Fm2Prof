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
