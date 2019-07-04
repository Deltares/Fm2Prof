import unittest, pytest
import sys, os

import shutil

import TestUtils
from fm2prof.main import Fm2ProfRunner, IniFile

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
    Create test output directory so it's easier to collect all output afterwards.
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
    Helper to split to set up an output directory for the generated data of each test case.
    """
    output_directory = base_output_dir + "\\{0}".format(caseName)

    # clean up the test case output directory if it is no empty
    if os.path.exists(output_directory) and os.listdir(output_directory):
        shutil.rmtree(output_directory)
    
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    return output_directory
# endregion   

@pytest.mark.integrationtest
def test_when_no_file_path_then_no_exception_is_risen():
    """ 1. Set up initial test data """
    iniFilePath = ''

    """ 2. Run test """
    try:
        fm2ProfRunner = Fm2ProfRunner(iniFilePath)
    except:
        pytest.fail('No exception expected.') 
        
@pytest.mark.integrationtest
def test_given_inifile_then_no_exception_is_risen():
    #1. Set up initial test data
    ini_file_name = 'fm2prof.ini'
    test_data_dir = TestUtils.get_test_data_dir('IniFile')
    ini_file_path = os.path.join(test_data_dir, ini_file_name)
    
    #2. Verify the initial expectations
    assert os.path.exists(ini_file_path), "Test File {} was not found".format(ini_file_path)
    
    #3. Run test
    try:
        fm2ProfRunner = Fm2ProfRunner(ini_file_path)
    except:
        pytest.fail('No exception expected.') 
