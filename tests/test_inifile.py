import unittest
import pytest
import sys
import os
import numbers

import shutil

from tests.TestUtils import TestUtils
from fm2prof.IniFile import IniFile

_root_output_dir = None


class Test_IniFile:
    @pytest.mark.unittest
    def test_when_no_file_path_then_no_exception_is_risen(self):
        # 1. Set up initial test data
        iniFilePath = ''

        # 2. Run test
        try:
            IniFile(iniFilePath)
        except:
            pytest.fail('No exception expected.')

    @pytest.mark.unittest
    def test_when_non_existent_file_path_then_io_exception_is_risen(self):
        # 1. Set up initial test data
        ini_file_path = 'nonexistent_ini_file.ini'

        # 2. Set expectations
        expected_error = '' + \
            'The given file path {} could not be found.'.format(ini_file_path)

        # 3. Run test
        with pytest.raises(IOError) as e_info:
            IniFile(ini_file_path)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, '' + \
            'Expected exception message {},'.format(expected_error) + \
            'retrieved {}'.format(error_message)


class Test_extract_input_parameters:

    _test_scenarios_input_parameters = [
        ('42', 42),
        ('4.2', 4.2),
        ('', None),
        ('dummy', None),
    ]

    @pytest.mark.unittest
    def test_when_no_input_parameters_key_returns_none(self):
        # 1. Set up initial test data
        iniFilePath = None
        iniFile = IniFile(iniFilePath)
        inifile_params = {'DummyKey': {}}
        new_params = None

        # 2. Run test
        try:
            new_params = iniFile._extract_input_parameters(inifile_params)
        except:
            pytest.fail('Test failed while trying to extract parameters.')

        # 3. Verify final expectations
        assert new_params is None

    @pytest.mark.unittest
    def test_when_no_parameters_no_exception_is_risen(self):
            # 1. Set up initial test data
        iniFilePath = None
        iniFile = IniFile(iniFilePath)
        inifile_parameters = None

        # 2. Run test """
        try:
            iniFile._extract_input_parameters(inifile_parameters)
        except:
            pytest.fail('Test failed while trying to extract parameters.')

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "value_as_string, expected_value",
        _test_scenarios_input_parameters)
    def test_when_parameters_are_given_then_maps_as_expected(
            self, value_as_string, expected_value):
        # 1. Set up initial test data
        iniFilePath = None
        iniFile = IniFile(iniFilePath)
        parameter_name = 'dummy_value'
        parameter_list = {parameter_name: value_as_string}
        inifile_parameters = {'InputParameters': parameter_list}
        new_parameters = None

        # 2. Verify initial expectations
        assert value_as_string != expected_value

        # 3. Run test
        try:
            new_parameters = iniFile._extract_input_parameters(
                inifile_parameters)
        except:
            pytest.fail('Test failed while trying to extract parameters.')

        # 4. Verify final expectations
        assert new_parameters is not None
        retievedValue = new_parameters[parameter_name]
        assert retievedValue == expected_value
        assert isinstance(retievedValue, numbers.Integral) == \
            isinstance(expected_value, numbers.Integral)

    @pytest.mark.unittest
    def test_when_no_input_parameters_key_returns_empty_dict(self):
        # 1. Set up initial test data
        iniFilePath = None
        iniFile = IniFile(iniFilePath)
        inifile_parameters = {'DummyKey': {}}
        new_parameters = None

        # 2. Run test
        try:
            new_parameters = iniFile._extract_input_files(inifile_parameters)
        except:
            pytest.fail('Test failed while trying to extract parameters.')

        # 3. Verify final expectations
        assert new_parameters is not None
        assert new_parameters == {}


class Test_extract_input_files:
    @pytest.mark.unittest
    def test_when_given_correct_parameters_then_gets_filenames(self):
        # 1. Set up initial test data
        iniFilePath = None
        new_parameters = None
        iniFile = IniFile(iniFilePath)

        file_name = 'Dummy_File_Path'
        file_key = 'dummy_file_path'
        file_value = 'dummyValue'

        parameter_list = {file_name: file_value}
        inifile_parameters = {'InputFiles': parameter_list}

        # 2. Run test
        try:
            new_parameters = iniFile._extract_input_files(inifile_parameters)
        except:
            pytest.fail('Test failed while trying to extract parameters.')

        # 3. Verify final expectations
        assert not(file_name in new_parameters)
        assert file_key in new_parameters
        assert new_parameters[file_key] == file_value


class Test_extract_output_dir:
    _test_scenarios_output_dir_cases = [
        (None, None, 'CaseName01'),
        ('', '', 'CaseName01'),
        ('dummyDir', '', 'dummyDir/CaseName01'),
        (None, 'dummycase', 'dummycase01'),
        ('dummydir', 'dummycase', 'dummydir/dummycase01'),
    ]

    @pytest.mark.unittest
    def test_when_no_outputdir_key_then_returns_none(self):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        new_output_dir_value = None
        inifile_parameters = {'DummykKey': {}}

        # 2. Run test
        try:
            new_output_dir_value = iniFile._extract_output_dir(
                inifile_parameters)
        except:
            pytest.fail('Test failed while trying to extract parameters.')

        # 3. Verify final expectations
        assert new_output_dir_value is None

    @pytest.mark.integrationtest
    @pytest.mark.parametrize(
        "param_output_dir_value, param_case_name_value, expected_value",
        _test_scenarios_output_dir_cases)
    def test_when_given_parameters_then_gets_correct_path(
            self,
            param_output_dir_value, param_case_name_value, expected_value):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        new_output_dir_value = None

        output_dir_value = param_output_dir_value
        case_name_value = param_case_name_value

        parameter_list = {
            'outputdir': output_dir_value,
            'casename': case_name_value
        }
        inifile_parameters = {'OutputDirectory': parameter_list}

        # 2. Run test
        try:
            new_output_dir_value = iniFile._extract_output_dir(
                inifile_parameters)
        except:
            pytest.fail('Test failed while trying to get a vlid output dir.')

        # 3. Verify final expectations
        assert expected_value in new_output_dir_value

    @pytest.mark.integrationtest
    def test_when_dirs_exist_then_returns_new_values(self):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        repeated_iterations = 5
        case_name = 'dummyCaseName'
        output_dir = 'IniFileTests'
        output_dir_list = []
        output_parameters = {
            'outputdir': output_dir,
            'casename': case_name
        }

        inifile_parameters = {
            'OutputDirectory': output_parameters
        }

        # 2. Verify initial expectations
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        # 3. Run test
        try:
            for idx in range(0, repeated_iterations):
                new_output_dir = iniFile._extract_output_dir(
                    inifile_parameters)
                os.makedirs(new_output_dir)
                output_dir_list.append(new_output_dir)
        except:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            pytest.fail('Test failed while trying to get valid case names.')

        # 4. Verify final expectations
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        assert output_dir_list
        set_output_dir_list = set(output_dir_list)
        assert len(output_dir_list) == repeated_iterations
        assert len(set_output_dir_list) == repeated_iterations


class Test_gets_valid_case_name:
    _test_scenarios_case_names = [
        ('', '', 'CaseName01'),
        (None, None, 'CaseName01'),
        ('dummyCase', None, 'dummyCase01'),
        (None, 'dummyDir', 'CaseName01'),
        ('dummyCase', 'dummyDir', 'dummyCase01')
    ]

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "case_name, output_dir, expected_value",
        _test_scenarios_case_names)
    def test_when_given_valid_parameters_then_returns_expected_values(
            self, case_name, output_dir, expected_value):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        new_case_name = None

        # 2. Run test
        try:
            new_case_name = iniFile._get_valid_case_name(case_name, output_dir)
        except:
            pytest.fail('Test failed while trying to extract parameters.')

        # 3. Verify final expectations
        assert new_case_name == expected_value

    @pytest.mark.integrationtest
    def test_when_dirs_exist_then_returns_new_valid_case_name(self):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        repeated_iterations = 5
        case_name = 'dummyCaseName'
        output_dir = 'IniFileTests'
        case_names = []

        # 2. Verify initial expectations
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        # 3. Run test
        try:
            for idx in range(0, repeated_iterations):
                new_case_name = iniFile._get_valid_case_name(
                    case_name, output_dir)
                relative_dir = os.path.join(output_dir, new_case_name)
                os.mkdir(relative_dir)
                case_names.append(new_case_name)
        except:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            pytest.fail('Test failed while trying to get valid case names.')

        # 4. Verify final expectations
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        assert case_names
        assert len(case_names) == repeated_iterations
        set_case_names = set(case_names)
        assert len(set_case_names) == repeated_iterations


class Test_get_valid_output_dir:
    _test_scenarios_output_dirs = [
        ('dummydir', 'dummydir'),
        ('dummydir/dummysubdir', 'dummydir\\dummysubdir'),
        ('../dummysubdir', '\\dummysubdir'),
    ]

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "output_dir, expected_value",
        _test_scenarios_output_dirs)
    def test_when_given_valid_parameters_then_returns_expected_values(
            self, output_dir, expected_value):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        new_output_dir = None

        # 2. Run test
        try:
            new_output_dir = iniFile._get_valid_output_dir(output_dir)
        except:
            err_mssg = 'Test failed while trying to get new valid output dir.'
            pytest.fail(err_mssg)

        # 3. Verify final expectations
        assert expected_value in new_output_dir

    @pytest.mark.unittest
    def test_when_no_output_dir_then_returns_current_path(self):
        # 1. Set initial test data
        ini_file_path = None
        iniFile = IniFile(ini_file_path)
        new_output_dir = None
        expected_value = os.getcwd()

        # 2. Run test
        try:
            new_output_dir = iniFile._get_valid_output_dir(None)
        except:
            err_mssg = 'Test failed while trying to get new valid output dir.'
            pytest.fail(err_mssg)

        # 3. Verify final expectations
        assert expected_value == new_output_dir


class Test_readini_file:

    @pytest.mark.unittest
    def test_when_no_file_path_then_io_exception_is_risen(self):
        # 1. Set up initial test data """
        iniFilePath = ''
        iniFile = IniFile(iniFilePath)

        # 2. Set expectations
        expected_error = 'No ini file was specified and no data could be read.'

        # 3. Run test
        with pytest.raises(IOError) as e_info:
            iniFile._read_inifile(iniFilePath)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, '' + \
            'Expected exception message {},'.format(expected_error) + \
            'retrieved {}'.format(error_message)

    @pytest.mark.systemtest
    def test_when_inifile_contains_output_dir_then_sets_output_dir(self):
        # 1. Set initial test data
        test_data_dir = TestUtils.get_local_test_data_dir('IniFile')
        ini_file_path = os.path.join(test_data_dir, 'valid_ini_file.ini')
        output_dir = 'tmp\\dummyTest01'
        expected_output_dir = os.path.join(os.getcwd(), output_dir)

        # 2. Verify initial expectations
        assert os.path.exists(ini_file_path), "Test file was not found."

        # 3. Run test
        try:
            ini_file = IniFile(ini_file_path)
        except:
            pytest.fail('Test failed while reading an ini file.')

        # 3. Verify final expectations
        assert ini_file is not None
        assert ini_file._output_dir is not None
        assert ini_file._output_dir != ''

        format_output_dir = ini_file._output_dir.replace('/', '\\')
        assert output_dir in format_output_dir
        assert format_output_dir == expected_output_dir

    @pytest.mark.systemtest
    def test_when_inifile_contains_input_parameters_then_sets_them(self):
        # 1. Set initial test data
        test_data_dir = TestUtils.get_local_test_data_dir('IniFile')
        ini_file_path = os.path.join(test_data_dir, 'valid_ini_file.ini')
        expected_input_parameters = {
            'number_of_css_points': 20,
            'transitionheight_sd': 0.25,
            'velocity_threshold': 0.01,
            'relative_threshold': 0.03,
            'min_depth_storage': 0.02,
            'plassen_timesteps': 10,
            'storagemethod_wli': 1,
            'bedlevelcriterium': 0.0,
            'sdstorage': 1,
            'frictionweighing': 0,
            'sectionsmethod': 1,
            'exportmapfiles': 0,
            'sdoptimisationmethod': 0,
        }

        # 2. Verify initial expectations
        assert os.path.exists(ini_file_path), "Test file was not found."

        # 3. Run test
        try:
            ini_file = IniFile(ini_file_path)
        except:
            pytest.fail('Test failed while reading an ini file.')

        # 4. Verify final expectations
        assert ini_file is not None
        assert ini_file._input_parameters is not None
        for expected_input_param in expected_input_parameters:
            expected_value = expected_input_parameters[expected_input_param]
            read_param_value = ini_file._input_parameters.get(
                expected_input_param)

            assert read_param_value is not None, '' + \
                'Key {} was not read or '.format(expected_input_param) + \
                'could not be found in {}'.format(ini_file._input_parameters)

            assert read_param_value == expected_value, '' + \
                'Expected value does not match ' + \
                'for key {}'.format(expected_input_param)

    @pytest.mark.systemtest
    def test_when_inifile_contains_input_files_then_sets_input_file_paths(
            self):
        # 1. Set initial test data
        test_data_dir = TestUtils.get_local_test_data_dir('IniFile')
        ini_file_path = os.path.join(test_data_dir, 'valid_ini_file.ini')
        expected_input_files = {
            'fm_netcdfile': 'dummy_file.nc',
            'crosssectionlocationfile': 'dummy_file.xyz',
            'regionpolygonfile': 'dummy_file.geojson',
            'sectionpolygonfile': 'dummy_file.geojson',
        }

        # 2. Verify initial expectations
        assert os.path.exists(ini_file_path), "Test file was not found."

        # 3. Run test
        try:
            ini_file = IniFile(ini_file_path)
        except:
            pytest.fail('Test failed while reading an ini file.')

        # 4. Verify final expectations
        assert ini_file is not None
        assert ini_file._input_parameters is not None

        for expected_input_file in expected_input_files:
            expected_value = expected_input_files[expected_input_file]
            read_param_value = ini_file._input_file_paths.get(
                expected_input_file)
            assert read_param_value is not None, '' + \
                'Key {} was not read or could not be found in {}'.format(
                    expected_input_file, ini_file._input_file_paths)
            assert read_param_value == expected_value, '' + \
                'Expected value does not match for key {}'.format(
                    expected_input_file)

    @pytest.mark.systemtest
    def test_when_inifile_is_not_valid_then_raises_exception(self):
        # 1. Set initial test data
        test_data_dir = TestUtils.get_local_test_data_dir('IniFile')
        ini_file_path = os.path.join(test_data_dir, 'invalid_ini_file.ini')
        ini_file = IniFile(None)

        # 2. Verify initial expectations
        assert ini_file is not None
        assert os.path.exists(ini_file_path), 'Test file was not found.'
        expected_error = 'It was not possible to extract ini parameters ' + \
            'from the file {}.'.format(ini_file_path)

        # 3. Run test
        with pytest.raises(Exception) as e_info:
            ini_file._read_inifile(ini_file_path)

        # 4. Verify final expectations
        assert ini_file is not None
        assert expected_error in str(e_info.value)


class Test_get_inifile_params:

    @pytest.mark.unittest
    def test_when_no_file_path_then_no_except_risen_and_returns_empty_dict(
            self):
        # 1. Set up initial test data """
        ini_file_path = ''
        return_value = None

        # 2. Set up expecations
        expected_return_value = {}

        # 3. Run test
        try:
            return_value = IniFile.get_inifile_params(ini_file_path)
        except Exception as e_info:
            pytest.fail('No exception expected but was thrown: ' +
                        '{}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value == expected_return_value

    @pytest.mark.unittest
    def test_when_given_invalid_inifile_then_exception_is_risen(self):
        # 1. Set up initial test data """
        test_data_dir = TestUtils.get_local_test_data_dir('IniFile')
        ini_file_path = os.path.join(test_data_dir, 'invalid_ini_file.ini')
        return_value = None

        # 2. Set up expecations
        expected_return_value = None

        # 3. Run test
        with pytest.raises(Exception):
            return_value = IniFile.get_inifile_params(ini_file_path)

        # 4. Verify final expectations
        assert return_value == expected_return_value

    @pytest.mark.systemtest
    def test_when_given_valid_inifile_then_gets_expected_values(self):
        # 1. Set initial test data
        test_data_dir = TestUtils.get_local_test_data_dir('IniFile')
        ini_file_path = os.path.join(test_data_dir, 'valid_dummy_inifile.ini')
        return_dict = None

        # 2. Set initial expectations
        assert os.path.exists(ini_file_path), "Test file was not found."
        expected_section_zero = {
            'option_zero_zero': 'string',
            'option_zero_one': '4.2'}
        expected_section_one = {
            'option_one_zero': '42',
            'option_one_two': 'string'}
        expected_section_two = {}
        expected_dict = {
            'section_zero': expected_section_zero,
            'section_one': expected_section_one,
            'section_two': expected_section_two
        }

        # 3. Run test
        try:
            return_dict = IniFile.get_inifile_params(ini_file_path)
        except:
            pytest.fail('Test failed while reading an ini file.')

        # 4. Verify final expectations
        assert return_dict is not None
        assert return_dict == expected_dict, '' + \
            'Expected {},'.format(expected_dict) + \
            ' but returned {}'.format(return_dict)
