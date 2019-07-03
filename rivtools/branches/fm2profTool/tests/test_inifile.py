import unittest
import pytest
import sys
import os

import shutil
import TestUtils

from fm2prof.main import IniFile

_root_output_dir = None


@pytest.mark.unittest
def test_IniFile_When_No_FilePath_Then_No_Exception_Is_Risen():
    """ 1. Set up initial test data """
    iniFilePath = ''

    """ 2. Run test """
    try:
        iniFile = IniFile(iniFilePath)
    except:
        pytest.fail('No exception expected.')


@pytest.mark.unittest
def test_IniFile_ReadIniFile_When_No_FilePath_Then_No_Exception_Is_Risen():
    """ 1. Set up initial test data """
    iniFilePath = ''
    iniFile = IniFile(iniFilePath)

    """ 2. Run test """
    with pytest.raises(Exception) as e_info:
        iniFile._read_inifile(iniFilePath)


@pytest.mark.unittest
def test_IniFile_extract_input_parameters_When_No_Input_Parameters_Key_Returns_None():
    # 1. Set up initial test data
    iniFilePath = None
    iniFile = IniFile(iniFilePath)
    inifile_parameters = {'DummyKey': {}}
    new_parameters = None

    # 2. Run test
    try:
        new_parameters = iniFile._extract_input_parameters(inifile_parameters)
    except:
        pytest.fail('Test failed while trying to extract parameters.')

    # 3. Verify final expectations
    assert new_parameters is None

@pytest.mark.unittest
def test_IniFile_extract_input_parameters_When_No_Parameters_No_Exception_Is_Risen():
        # 1. Set up initial test data
    iniFilePath = None
    iniFile = IniFile(iniFilePath)
    inifile_parameters = None

    # 2. Run test """
    try:
        iniFile._extract_input_parameters(inifile_parameters)
    except:
        pytest.fail('Test failed while trying to extract parameters.')


_test_scenarios_input_parameters = [
    ('42', 42),
    ('4.2', 4.2),
    ('', None),
    ('dummy', None),
]

@pytest.mark.unittest
@pytest.mark.parametrize("value_as_string, expected_value", _test_scenarios_input_parameters)
def test_IniFile_extract_input_parameters_When_Parameters_Are_Given_Then_Maps_As_Expected(value_as_string, expected_value):
    # 1. Set up initial test data
    iniFilePath = None
    iniFile = IniFile(iniFilePath)
    parameter_name = 'dummy_value'
    parameter_list = {parameter_name : value_as_string}
    inifile_parameters = {'InputParameters': parameter_list}
    new_parameters = None

    # 2. Verify initial expectations
    assert value_as_string != expected_value

    # 3. Run test
    try:
        new_parameters = iniFile._extract_input_parameters(inifile_parameters)
    except:
        pytest.fail('Test failed while trying to extract parameters.')

    # 4. Verify final expectations
    assert new_parameters is not None
    assert new_parameters[parameter_name] == expected_value

@pytest.mark.unittest
def test_IniFile_extract_input_files_When_No_Input_Parameters_Key_Returns_EmptyDict():
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

@pytest.mark.unittest
def test_IniFile_extract_input_files_gets_filenames():
    # 1. Set up initial test data
    iniFilePath = None
    new_parameters = None
    iniFile = IniFile(iniFilePath)
    
    file_name = 'Dummy_File_Path'
    file_key = 'dummy_file_path'
    file_value = 'dummyValue'
    
    parameter_list = {file_name : file_value}
    inifile_parameters = {'InputFiles': parameter_list}

    # 2. Run test
    try:
        new_parameters = iniFile._extract_input_files(inifile_parameters)
    except:
        pytest.fail('Test failed while trying to extract parameters.')

    # 3. Verify final expectations
    assert not( file_name in new_parameters)
    assert file_key in new_parameters
    assert new_parameters[file_key] == file_value

@pytest.mark.unittest
def test_IniFile_extract_output_dir_no_outputdir_key_returns_None():
    # 1. Set initial test data
    ini_file_path = None
    iniFile = IniFile(ini_file_path)
    new_output_dir_value = None
    inifile_parameters = {'DummykKey': {}}
    
    # 2. Run test
    try:
        new_output_dir_value = iniFile._extract_output_dir(inifile_parameters)
    except:
        pytest.fail('Test failed while trying to extract parameters.')
    
    # 3. Verify final expectations
    assert new_output_dir_value is None

_test_scenarios_output_dir_cases = [
    (None,None, 'CaseName01'),
    ('','', 'CaseName01'),
    ('dummyDir','', 'dummyDir/CaseName01'),
    (None,'dummycase', 'dummycase01'),
    ('dummydir','dummycase', 'dummydir/dummycase01'),
]

@pytest.mark.integrationtest
@pytest.mark.parametrize("param_output_dir_value, param_case_name_value, expected_value", _test_scenarios_output_dir_cases)
def test_IniFile_extract_output_dir_gets_correct_path(param_output_dir_value, param_case_name_value, expected_value):
    # 1. Set initial test data
    ini_file_path = None
    iniFile = IniFile(ini_file_path)
    new_output_dir_value = None
    
    output_dir_value = param_output_dir_value
    case_name_value = param_case_name_value
    
    parameter_list = {
        'outputdir' : output_dir_value,
        'casename'  : case_name_value
    }
    inifile_parameters = {'OutputDirectory': parameter_list}
    
    # 2. Run test
    try:
        new_output_dir_value = iniFile._extract_output_dir(inifile_parameters)
    except:
        pytest.fail('Test failed while trying to get a vlid output dir.')
    
    # 3. Verify final expectations
    assert expected_value in new_output_dir_value


_test_scenarios_case_names = [
    ('','', 'CaseName01'),
    (None, None, 'CaseName01'),
    ('dummyCase', None, 'dummyCase01'),
    (None, 'dummyDir', 'CaseName01'),
    ('dummyCase', 'dummyDir', 'dummyCase01')
]

@pytest.mark.unittest
@pytest.mark.parametrize("case_name, output_dir, expected_value", _test_scenarios_case_names)
def test_IniFile_get_valid_case_name_returns_expected_values(case_name, output_dir, expected_value):
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


_test_scenarios_output_dirs = [
    ('dummydir', 'dummydir'),
    ('dummydir/dummysubdir', 'dummydir\\dummysubdir'),
    ('../dummysubdir', '\\dummysubdir'),
]

@pytest.mark.unittest
@pytest.mark.parametrize("output_dir, expected_value", _test_scenarios_output_dirs)
def test_IniFile_get_valid_output_dir_returns_expected_values(output_dir, expected_value):
    # 1. Set initial test data
    ini_file_path = None
    iniFile = IniFile(ini_file_path)
    new_output_dir = None
    
    # 2. Run test
    try:
        new_output_dir = iniFile._get_valid_output_dir(output_dir)
    except:
        pytest.fail('Test failed while trying to get new valid output dir.')
    
    # 3. Verify final expectations
    assert expected_value in new_output_dir

@pytest.mark.unittest
def test_IniFile_get_valid_output_dir_when_no_output_dir_returns_current_path():
    # 1. Set initial test data
    ini_file_path = None
    iniFile = IniFile(ini_file_path)
    new_output_dir = None
    expected_value = os.getcwd()
    
    # 2. Run test
    try:
        new_output_dir = iniFile._get_valid_output_dir(None)
    except:
        pytest.fail('Test failed while trying to get new valid output dir.')
    
    # 3. Verify final expectations
    assert expected_value == new_output_dir

@pytest.mark.systemtest
def test_IniFile_read_ini_file_sets_output_dir():
    # 1. Set initial test data
    test_data_dir = TestUtils.get_test_data_dir('IniFile')
    ini_file_path = os.path.join(test_data_dir, 'valid_ini_file.ini')
    output_dir = 'tmp\\dummyTest01'
    expected_output_dir = os.path.join( os.getcwd(), output_dir )


    # 2. Verify initial expectations
    assert os.path.exists(ini_file_path), "Test file was not found."
    
    # 3. Run test
    try:
        ini_file = IniFile(ini_file_path)
    except:
        pytest.fail('Test failed while reading an ini file.')
    
    # 3. Verify final expectations
    assert ini_file != None
    assert ini_file._output_dir != None
    assert ini_file._output_dir != ''
    
    format_output_dir = ini_file._output_dir.replace('/','\\')
    assert output_dir in format_output_dir
    assert format_output_dir == expected_output_dir

@pytest.mark.systemtest
def test_IniFile_read_ini_file_sets_input_parameters():
    # 1. Set initial test data
    test_data_dir = TestUtils.get_test_data_dir('IniFile')
    ini_file_path = os.path.join(test_data_dir, 'valid_ini_file.ini')
    expected_input_parameters = {
        'number_of_css_points'  : 20,
        'transitionheight_sd'   : 0.25,
        'velocity_threshold'    : 0.01,
        'relative_threshold'    : 0.03,
        'min_depth_storage'     : 0.02,
        'plassen_timesteps'     : 10,
        'storagemethod_wli'     : 1,
        'bedlevelcriterium'     : 0.1,
        'sdstorage'             : 1,
        'frictionweighing'      : 0,
        'sectionsmethod'        : 1,
    }

    # 2. Verify initial expectations
    assert os.path.exists(ini_file_path), "Test file was not found."
    
    # 3. Run test
    try:
        ini_file = IniFile(ini_file_path)
    except:
        pytest.fail('Test failed while reading an ini file.')
    
    # 4. Verify final expectations
    assert ini_file != None
    assert ini_file._input_parameters != None
    for expected_input_param in expected_input_parameters:
        expected_value = expected_input_parameters[expected_input_param]
        read_param_value = ini_file._input_parameters.get(expected_input_param)
        assert read_param_value is not None, 'Key {} was not read or could not be found in {}'.format(expected_input_param, ini_file._input_parameters)
        assert read_param_value == expected_value, 'Expected value does not match for key {}'.format(expected_input_param)

@pytest.mark.systemtest
@pytest.mark.developtest
def test_IniFile_read_ini_file_sets_input_file_paths():
    # 1. Set initial test data
    test_data_dir = TestUtils.get_test_data_dir('IniFile')
    ini_file_path = os.path.join(test_data_dir, 'valid_ini_file.ini')
    expected_input_files = {
        'fm_netcdfile'              : 'dummy_file.nc',
        'crosssectionlocationfile'  : 'dummy_file.xyz',
        'gebiedsvakken'             : 'dummy_file.xyz',
        'sectionfractionfile'       : 'dummy_file.txt',
    }

    # 2. Verify initial expectations
    assert os.path.exists(ini_file_path), "Test file was not found."
    
    # 3. Run test
    try:
        ini_file = IniFile(ini_file_path)
    except:
        pytest.fail('Test failed while reading an ini file.')
    
    # 4. Verify final expectations
    assert ini_file != None
    assert ini_file._input_parameters != None
    for expected_input_file in expected_input_files:
        expected_value = expected_input_files[expected_input_file]
        read_param_value = ini_file._input_file_paths.get(expected_input_file)
        assert read_param_value is not None, 'Key {} was not read or could not be found in {}'.format(expected_input_file, ini_file._input_file_paths)
        assert read_param_value == expected_value, 'Expected value does not match for key {}'.format(expected_input_file)