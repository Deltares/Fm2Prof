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
