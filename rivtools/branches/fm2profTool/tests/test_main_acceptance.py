import unittest, pytest
import sys, os

import shutil

import TestUtils
from fm2prof.main import Fm2ProfRunner

@pytest.mark.acceptance
def test_run_case_01():
    caseName = "case_01_rectangle"
    
    # 1. Set up test data.
    directory = TestUtils.get_external_test_data_dir(caseName)       
    map_file = directory + 'Data\\FM\\50x25_mesh\\FlowFM_fm2prof_map.nc'
    css_file = directory + 'Data\\cross_section_locations.xyz'
    chainage_file = directory + 'Data\\cross_section_chainages.txt'

    output_directory = __check_and_create_test_case_output_dir(caseName)

    # Create the runner and set the saving figures variable to true
    runner = Fm2ProfRunner(output_directory, saveFigures=True)

    # 2. Verify precondition (no output generated)
    assert os.path.exists(output_directory) and not os.listdir(output_directory)

    # 3. Run file:
    runner.run_with_files(map_file, css_file, chainage_file)

    # 4. Verify there is output generated:
    assert os.listdir(output_directory), "There is no output generated for {0}".format(caseName)

def __check_and_create_test_root_output_dir():
    """
    Create test output directory so it's easier to collect all output afterwards.
    """
    rootOutputDir = os.path.join(os.path.dirname(__file__), "Output")
    if not os.path.exists(rootOutputDir):
        os.mkdir(rootOutputDir)
    return rootOutputDir

def __check_and_create_test_case_output_dir(caseName):
    """
    Helper to split to set up an output directory for the generated data of each test case.
    """
    output_directory = __check_and_create_test_root_output_dir() + "\\{0}".format(caseName)

    # clean up the test case output directory if it is no empty
    if os.path.exists(output_directory) and os.listdir(output_directory):
        shutil.rmtree(output_directory)
    
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    return output_directory
# endregion    