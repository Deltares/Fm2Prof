import unittest, pytest
import sys, os

import shutil

import TestUtils
from fm2prof.Fm2ProfRunner import Fm2ProfRunner
from fm2prof.IniFile import IniFile

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

""" To use excluding markups the following command line can be used:
- Include only tests that are acceptance that ARE NOT slow:
    pytest tests -m "acceptance and not slow"
- Include only tests that are both acceptance AND slow:
    pytest tests -m "acceptance and slow"
 """

_test_scenarios = [
    pytest.param('case_01_rectangle', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz'),
    pytest.param('case_02_compound', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz'),
    pytest.param('case_03_threestage', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz'),
    pytest.param('case_04_storage', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz'),
    pytest.param('case_05_dyke', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz'),
    pytest.param('case_06_plassen', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz'),
    pytest.param('case_07_triangular', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz'),
    pytest.param('case_08_waal', 'Data\\FM\\FlowFM_fm2prof_map.nc', 'Data\\cross_section_locations.xyz', marks = pytest.mark.slow)
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

class Test_Fm2Prof_Run_IniFile:

    @pytest.mark.acceptance
    @pytest.mark.timeout(7200)
    @pytest.mark.parametrize(("case_name", "map_file", "css_file"), _test_scenarios, ids=_test_scenarios_ids)
    def test_when_given_input_data_then_output_is_generated(self, case_name, map_file, css_file):               
        # 1. Set up test data.
        iniFilePath = None
        iniFile = IniFile(iniFilePath)
        test_data_dir = TestUtils.get_external_test_data_dir(case_name)
        iniFile._output_dir = __check_and_create_test_case_output_dir(__get_base_output_dir(), case_name)
        iniFile._input_file_paths = {
            "fm_netcdfile": os.path.join(test_data_dir, map_file),
            'crosssectionlocationfile' : os.path.join(test_data_dir, css_file),
        }
        iniFile._input_parameters = {
            "number_of_css_points"  :	20,        
            "transitionheight_sd"	:	0.25,
            "velocity_threshold"	:	0.01,	
            "relative_threshold"	:	0.03,	
            "min_depth_storage"	    :	0.02,	
            "plassen_timesteps"	    :	10,	
            "storagemethod_wli"	    :	1,		
            "bedlevelcriterium"	    :	0.1,
            "sdstorage"			    :	1,	
            "frictionweighing"	    :	0,		
            "sectionsmethod"		:	0		
        }

        # Create the runner and set the saving figures variable to true
        runner = Fm2ProfRunner(iniFilePath)

        # 2. Verify precondition (no output generated)
        assert os.path.exists(iniFile._output_dir) and not os.listdir(iniFile._output_dir)

        # 3. Run file:
        runner.run_inifile(iniFile)

        # 4. Verify there is output generated:
        assert os.listdir(iniFile._output_dir), "There is no output generated for {0}".format(case_name)

class Test_Main_Run_IniFile:
    
    def __run_main_with_arguments(self, ini_file):
        pythonCall = "fm2prof\\main.py -i {0}".format(ini_file)
        os.system("python {0}".format(pythonCall))

    def __create_test_ini_file(self, root_dir, case_name, map_file, css_file):
        output_dir = os.path.join(root_dir, 'OutputFiles')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_files_key = 'InputFiles'
        input_parameters_key = 'InputParameters'
        output_directory_key = 'OutputDirectory'

        test_data_dir = TestUtils.get_external_test_data_dir(case_name)
        input_file_paths = {
            "fm_netcdfile": os.path.join(test_data_dir, map_file),
            'crosssectionlocationfile' : os.path.join(test_data_dir, css_file),
        }
        input_parameters = {
            "number_of_css_points"  :	20,        
            "transitionheight_sd"	:	0.25,
            "velocity_threshold"	:	0.01,	
            "relative_threshold"	:	0.03,	
            "min_depth_storage"	    :	0.02,	
            "plassen_timesteps"	    :	10,	
            "storagemethod_wli"	    :	1,		
            "bedlevelcriterium"	    :	0.1,
            "sdstorage"			    :	1,	
            "frictionweighing"	    :	0,		
            "sectionsmethod"		:	0		
        }

        # write file
        file_path = os.path.join(root_dir, '{}_ini_file.ini'.format(case_name))
        f = open(file_path, 'w+')
        
        f.writelines('[{}]\r\n'.format(input_files_key))
        for key, value in input_file_paths.items():
            f.writelines('{} = {}\r\n'.format(key, value))
        f.writelines('\r\n')
        f.writelines('[{}]\r\n'.format(input_parameters_key))
        for key, value in input_parameters.items():
            f.writelines('{} = {}\r\n'.format(key, value))
        
        f.writelines('\r\n')
        f.writelines('[{}]\r\n'.format(output_directory_key))
        f.writelines('OutputDir = {}\r\n'.format(output_dir))
        f.writelines('CaseName = {}\r\n'.format(case_name))
        
        f.close()
        return (file_path, output_dir)
    
    @pytest.mark.system
    def test_when_given_inifile_then_output_is_generated(self):               
        # 1. Set up test data.
        case_name = 'case_01_rectangle'
        map_file = 'Data\\FM\\FlowFM_fm2prof_map.nc'
        css_file = 'Data\\cross_section_locations.xyz'
        root_output_dir = os.path.join(os.path.dirname(__file__), "RunMainWithCustomIniFile", case_name)
        (ini_file_path, output_dir) = self.__create_test_ini_file(root_output_dir, case_name, map_file, css_file)

        # 2. Verify precondition (no output generated)
        assert os.path.exists(ini_file_path)
        expected_files = [
            'CrossSectionDefinitions.ini',
            'CrossSectionLocations.ini',
            'geometry.csv',
            'roughness.csv',
            'geometry_test.csv',
            'roughness_test.csv',
            'volumes.csv',
        ]

        # 3. Run file:
        try:
            self.__run_main_with_arguments(ini_file_path)
        except Exception as e_error:
            if os.path.exists(root_output_dir):
                shutil.rmtree(root_output_dir)
            pytest.fail('No exception expected but was thrown {}.'.format(str(e_error)))

        # 4. Verify there is output generated:
        output_files = os.path.join(output_dir, '{}01'.format(case_name))
        generated_files = os.listdir(output_files)
        if os.path.exists(root_output_dir):
            shutil.rmtree(root_output_dir)
        assert generated_files, "There is no output generated for {0}".format(case_name)
        for expected_file in expected_files:
            assert expected_file in generated_files
