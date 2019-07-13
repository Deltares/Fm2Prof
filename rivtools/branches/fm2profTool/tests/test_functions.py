import unittest
import pytest
import sys
import os
import numbers

import shutil

from tests.TestUtils import TestUtils
import fm2prof.Functions as Func


class Test_read_css_xyz:

    _test_scenarios_invalid_file_paths = [
        (None),
        (''),
        ('dummyFilePath')]

    @pytest.mark.unittest
    @pytest.mark.parametrize("file_path", _test_scenarios_invalid_file_paths)
    def test_read_css_xyz_given_nofilepath_then_no_exception_is_risen(
            self, file_path):
        # 1. Prepare test data
        return_value = None

        # 2. Run test
        with pytest.raises(IOError) as pytest_wrapped_e:
            return_value = Func._read_css_xyz(file_path)

        # 3. Verify final expectations
        assert return_value is None

    @pytest.mark.systemtest
    def test_read_css_xyz_valid_file_path_returns_expected_input_data(self):
        # 1. Prepare test data
        dir_name = 'functions_test_data'
        test_directory = TestUtils.get_local_test_data_dir(dir_name)
        file_name = 'cross_section_locations.xyz'
        file_path = os.path.join(test_directory, file_name)
        expected_input_data = {
            'xy': [(25.0, 75.0), (475.0, 75.0)],
            'id': ['case1_0', 'case1_500'],
            'length': [250.0, 500.0],
            'branchid': ['case1', 'case1'],
            'chainage': [0.0, 500.0]
        }

        # 2. Verify the initial expectation
        assert os.path.exists(file_path), "" + \
            "Test File {} could not be found".format(file_path)

        # 3. Run test
        try:
            result_input_data = Func._read_css_xyz(file_path)
        except:
            pytest.fail('No exception expected but test failed \
                while reading file {}.'.format(file_path))

        # 4. Verify final expectations
        for expected_input_key in expected_input_data:
            assert expected_input_key in result_input_data, '' + \
                'Key {} was not found in '.format(expected_input_key) + \
                'the result data {}'.format(result_input_data)

            result_input_value = result_input_data.get(expected_input_key)
            expected_input_value = expected_input_data.get(expected_input_key)
            assert result_input_value == expected_input_value, '' + \
                'Results did not match ' + \
                'for key {}, '.format(expected_input_key) + \
                'expected {}, '.format(expected_input_value) + \
                'result {}'.format(result_input_value)
