import numbers
import os
import shutil
import sys
import unittest
from pathlib import Path

import pytest

from fm2prof.Import import FMDataImporter, FmModelData
from tests.TestUtils import TestUtils, skipwhenexternalsmissing


class Test_FMDataImporter:
    @skipwhenexternalsmissing
    def test_when_map_file_without_czu_no_exception(self):
        # 1. Set up test data
        test_map = Path(TestUtils.get_local_test_data_dir("main_test_data")).joinpath(
            "fm_map.nc"
        )
        assert test_map.is_file()

        # 2. Set initial expectations
        return_value = FMDataImporter().import_dflow2d(test_map)

        # 3. Verify final expectations
        assert return_value is not None


class Test_FmModelData:
    @pytest.mark.parametrize("arg_list", [(""), (None)])
    def test_when_no_arguments_exception_is_risen(self, arg_list):
        # 1. Set up test data
        arg_list = arg_list

        # 2. Set initial expectations
        expected_error_message = "FM model data was not read correctly."

        # 3. Run test
        with pytest.raises(Exception) as pytest_wrapped_e:
            FmModelData(arg_list)

        # 4. Verify final expectations
        recieved_error_message = str(pytest_wrapped_e.value)
        assert expected_error_message == recieved_error_message, (
            ""
            + "Expected error message {},".format(expected_error_message)
            + " does not match generated {}".format(recieved_error_message)
        )

    @pytest.mark.parametrize("arg_list", [(""), (None)])
    def test_when_argument_length_not_as_expected_then_exception_is_risen(
        self, arg_list
    ):
        # 1. Set up test data
        arg_list = ["arg1", "arg2"]

        # 2. Set initial expectations
        expected_error_message = (
            ""
            + "Fm model data expects 5 arguments but only"
            + " {} were given".format(len(arg_list))
        )

        # 3. Run test
        with pytest.raises(Exception) as pytest_wrapped_e:
            FmModelData(arg_list)

        # 4. Verify final expectations
        recieved_error_message = str(pytest_wrapped_e.value)
        assert expected_error_message == recieved_error_message, (
            ""
            + "Expected error message {},".format(expected_error_message)
            + " does not match generated {}".format(recieved_error_message)
        )

    def test_when_given_expected_arguments_then_object_is_created(self):
        # 1. Set up test data
        time_dependent_data = "arg1"
        time_independent_data = "arg2"
        edge_data = "arg3"
        node_coordinates = "arg4"
        css_data = "arg5"
        arg_list = [
            time_dependent_data,
            time_independent_data,
            edge_data,
            node_coordinates,
            css_data,
        ]
        return_fm_model_data = None
        # 2. Run test
        try:
            return_fm_model_data = FmModelData(arg_list)
        except:
            pytest.fail("No exception expected but was thrown")

        # 4. Verify final expectations
        assert return_fm_model_data is not None
        assert return_fm_model_data.time_dependent_data == time_dependent_data
        assert return_fm_model_data.time_independent_data == time_independent_data
        assert return_fm_model_data.edge_data == edge_data
        assert return_fm_model_data.node_coordinates == node_coordinates
        assert return_fm_model_data.css_data_list == []

    def test_when_given_data_dictionary_then_css_data_list_is_set(self):
        # 1. Set up test data
        time_dependent_data = "arg1"
        time_independent_data = "arg2"
        edge_data = "arg3"
        node_coordinates = "arg4"
        dummy_key = "dummyKey"
        dummy_values = [0, 1]
        css_data_dict = {
            dummy_key: dummy_values,
        }

        arg_list = [
            time_dependent_data,
            time_independent_data,
            edge_data,
            node_coordinates,
            css_data_dict,
        ]
        return_fm_model_data = None

        # 2. Set expectations
        expected_css_data_list = [{dummy_key: 0}, {dummy_key: 1}]

        #  3. Run test
        try:
            return_fm_model_data = FmModelData(arg_list)
        except:
            pytest.fail("No exception expected but was thrown")

        # 4. Verify final expectations
        assert return_fm_model_data is not None
        assert return_fm_model_data.css_data_list != css_data_dict
        assert return_fm_model_data.css_data_list == expected_css_data_list


class Test_get_ordered_css_list:
    def test_when_given_dictionary_then_returns_list(self):
        # 1. Set up test_data
        return_list = None
        dummy_key = "dummyKey"
        dummy_values = [0, 1]
        test_dict = {
            dummy_key: dummy_values,
        }

        # 2. Run test
        expected_list = [{dummy_key: 0}, {dummy_key: 1}]

        # 3. Run test
        try:
            return_list = FmModelData.get_ordered_css_list(test_dict)
        except:
            pytest.fail("No exception expected but was thrown")

        # 4. Verify final expectations
        assert return_list is not None
        assert return_list == expected_list, (
            ""
            + "Expected return value {},".format(expected_list)
            + " but return {} instead.".format(return_list)
        )

    @pytest.mark.parametrize("test_dict", [(""), (None), ({})])
    def test_when_given_unexpected_value_then_returns_empty_list(self, test_dict):
        # 1. Set up test_data
        return_list = None
        dummy_key = "dummyKey"
        dummy_values = [0, 1]

        # 2. Run test
        expected_list = []

        # 3. Run test
        try:
            return_list = FmModelData.get_ordered_css_list(test_dict)
        except:
            pytest.fail("No exception expected but was thrown")

        # 4. Verify final expectations
        assert return_list is not None
        assert return_list == expected_list, (
            ""
            + "Expected return value {},".format(expected_list)
            + " but return {} instead.".format(return_list)
        )
