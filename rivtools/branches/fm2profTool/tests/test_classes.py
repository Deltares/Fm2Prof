import unittest
import pytest
import sys
import os
import numbers

import shutil
import TestUtils

import fm2prof.Classes as CE

class Test_FmModelData:
    @pytest.mark.unittest
    @pytest.mark.parametrize("arg_list", [(''), (None)])
    def test_when_no_arguments_exception_is_risen(self, arg_list):
        # 1. Set up test data
        arg_list = arg_list

        # 2. Set initial expectations
        expected_error_message = 'FM model data was not read correctly.'

        # 3. Run test
        with pytest.raises(Exception) as pytest_wrapped_e:
            CE.FmModelData(arg_list)
        
        # 4. Verify final expectations
        recieved_error_message = str(pytest_wrapped_e.value)
        assert expected_error_message == recieved_error_message, "Expected error message {}, does not match generated {}".format(expected_error_message, recieved_error_message)

    @pytest.mark.unittest
    @pytest.mark.parametrize("arg_list", [(''), (None)])
    def test_when_argument_length_not_as_expected_then_exception_is_risen(self, arg_list):
        # 1. Set up test data
        arg_list = ['arg1', 'arg2']

        # 2. Set initial expectations
        expected_error_message = 'Fm model data expects 5 arguments but only {} were given'.format(len(arg_list))

        # 3. Run test
        with pytest.raises(Exception) as pytest_wrapped_e:
            CE.FmModelData(arg_list)
        
        # 4. Verify final expectations
        recieved_error_message = str(pytest_wrapped_e.value)
        assert expected_error_message == recieved_error_message, "Expected error message {}, does not match generated {}".format(expected_error_message, recieved_error_message)

    @pytest.mark.unittest
    def test_when_parameters_match_expectations_then_object_is_created(self):
        # 1. Set up test data
        time_dependent_data = 'arg1'
        time_independent_data = 'arg2'
        edge_data = 'arg3'
        node_coordinates = 'arg4'
        css_data = 'arg5'
        arg_list = [time_dependent_data, time_independent_data, edge_data, node_coordinates, css_data]
        return_fm_model_data = None
        # 2. Run test
        try:
            return_fm_model_data = CE.FmModelData(arg_list)
        except:
            pytest.fail('No exception expected but was thrown')
        
        # 4. Verify final expectations
        assert return_fm_model_data is not None
        assert return_fm_model_data.time_dependent_data == time_dependent_data
        assert return_fm_model_data.time_independent_data == time_independent_data
        assert return_fm_model_data.edge_data == edge_data
        assert return_fm_model_data.node_coordinates == node_coordinates
        assert return_fm_model_data.css_data == css_data

