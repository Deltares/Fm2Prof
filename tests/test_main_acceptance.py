import unittest, pytest
import sys, os

import shutil

import TestUtils
from fm2prof.main import Fm2ProfRunner, IniFile

_root_output_dir = None

# Test data to be used
_test_scenarios_ids = [ 
    'case_01_rectangle',
    'case_02_compound',
    'case_03_threestage',
    'case_04_storage',
    'case_05_dyke', 
    'case_06_plassen',
    'case_07_triangular',
    'case_08_waal'
]

_test_scenarios = [
    ('case_01_rectangle', 'Data\\FM\\50x25_mesh\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt'),
    ('case_02_compound', 'Data\\FM\\50x25_mesh\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt'),
    ('case_03_threestage', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt'),
    ('case_04_storage', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt'),
    ('case_05_dyke', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt'),
    ('case_06_plassen', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt'),
    ('case_07_triangular', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt'),
    ('case_08_waal', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', 'Data\\cross_section_chainages.txt')
]

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

@pytest.mark.acceptance
@pytest.mark.parametrize("case_name, map_file, css_file, chainage_file", _test_scenarios, ids=_test_scenarios_ids)
def test_Fm2Prof_run_with_files(case_name, map_file, css_file, chainage_file):       
    
    # 1. Set up test data.
    iniFilePath = None
    iniFile = IniFile(iniFilePath)
    test_data_dir = TestUtils.get_external_test_data_dir(case_name)
    iniFile._output_dir = __check_and_create_test_case_output_dir(__get_base_output_dir(), case_name)
    iniFile._map_file = os.path.join(test_data_dir, map_file)
    iniFile._css_file = os.path.join(test_data_dir, css_file)
    iniFile._chainage_file = os.path.join(test_data_dir, chainage_file)
    iniFile._inputParam_dict = {
            "number_of_css_points"  :	20,        
            "transitionheight_sd"	:	0.25,
            "velocity_threshold"	:	0.01,	
            "relative_threshold"	:	0.03,	
            "min_depth_storage"	    :	0.02,	
            "plassen_timesteps"	    :	10,	
            "storagemethod_wli"	    :	1,		
            "bedlevelcriterium"	    :	0.1,
            "SDstorage"			    :	1,	
            "Frictionweighing"	    :	0,		
            "sectionsmethod"		:	1		
        }

    # Create the runner and set the saving figures variable to true
    runner = Fm2ProfRunner(iniFilePath)

    # 2. Verify precondition (no output generated)
    assert os.path.exists(iniFile._output_dir) and not os.listdir(iniFile._output_dir)

    # 3. Run file:
    runner.run_inifile(iniFile)

    # 4. Verify there is output generated:
    assert os.listdir(iniFile._output_dir), "There is no output generated for {0}".format(case_name)