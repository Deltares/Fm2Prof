import unittest, pytest
import sys, os
import datetime

import shutil

import TestUtils
from fm2prof.main import Fm2ProfRunner, IniFile
from fm2prof.Classes import CrossSection as CS
from fm2prof.Classes import FmModelData as FMD

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

class Test_Fm2ProfRunner:
    @pytest.mark.integrationtest
    def test_when_no_file_path_then_no_exception_is_risen(self):
        # 1. Set up initial test data
        iniFilePath = ''
        runner = None

        # 2. Run test
        try:
            runner = Fm2ProfRunner(iniFilePath)
        except Exception as e:
            pytest.fail('No exception expected, but thrown: {}'.format(str(e))) 
        
        # 3. Verify final expectations
        assert runner is not None
            
    @pytest.mark.integrationtest
    def test_given_inifile_then_no_exception_is_risen(self):
        #1. Set up initial test data
        ini_file_name = 'fm2prof.ini'
        test_data_dir = TestUtils.get_test_data_dir('IniFile')
        ini_file_path = os.path.join(test_data_dir, ini_file_name)
        runner = None

        #2. Verify the initial expectations
        assert os.path.exists(ini_file_path), "Test File {} was not found".format(ini_file_path)
        
        #3. Run test
        try:
            runner = Fm2ProfRunner(ini_file_path)
        except Exception as e:
            pytest.fail('No exception expected, but thrown: {}'.format(str(e))) 
        
        # 4. Verify final expectations
        assert runner is not None

class Test_generate_cross_section_list:
    
    @pytest.mark.unittest
    def test_when_not_given_FmModelData_then_returns_empty_list(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {
            'DummyKey' : 'dummyValue'
        }
        return_value = None
        
        # 2. Verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            return_value = runner._generate_cross_section_list(input_param_dict, None)
        except Exception as e_info:
            pytest.fail('Exception {} was given while generating cross sections'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is not None
        assert len(return_value) == 0
    
    @pytest.mark.unittest
    def test_when_not_given_input_param_dict_then_returns_empty_list(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        fm_model_data_args = (0,1,2,3,4)
        fm_model_data = FMD(fm_model_data_args)
        return_value = None
        
        # 2. Verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            return_value = runner._generate_cross_section_list(None, fm_model_data)
        except Exception as e_info:
            pytest.fail('Exception {} was given while generating cross sections'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is not None
        assert len(return_value) == 0
    
    @pytest.mark.integrationtest
    def test_when_given_correct_parameters_then_returns_list_with_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        fm_model_data_args = (0,1,2,3,4)
        fm_model_data = FMD(fm_model_data_args)
        input_param_dict = {
            'DummyKey' : 'dummyValue'
        }
        return_value = None
        
        # 2. Verify initial expectations
        assert runner is not None

        # 3. Run test
        try:
            return_value = runner._generate_cross_section_list(input_param_dict, fm_model_data)
        except Exception as e_info:
            pytest.fail('Exception {} was given while generating cross sections'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is not None
        assert len(return_value) == 0
        pytest.fail('To Do')

class Test_generate_cross_section:
    
    @pytest.mark.unittest
    def test_when_no_input_param_dict_is_given_then_expected_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        # 2. Set expectations
        expected_error = 'No input parameters (from ini file) given for new cross section {}'.format(test_css_name)

        # 3. Run test
        with pytest.raises(Exception) as e_info:
            runner._generate_cross_section(
                css_name = test_css_name, 
                input_param_dict = None, 
                fm_model_data = None)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, 'Expected exception message {}, retrieved {}'.format(expected_error, error_message)

    @pytest.mark.unittest
    def test_when_no_fm_model_data_is_given_then_expected_exception_risen(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        input_param_dict = {'dummyKey' : 'dummyValue'}

        # 2. Set expectations
        expected_error = 'No FM data given for new cross section {}'.format(test_css_name)

        # 3. Run test
        with pytest.raises(Exception) as e_info:
            runner._generate_cross_section(
                css_name = test_css_name, 
                input_param_dict = input_param_dict, 
                fm_model_data = None)

        # 4. Verify final expectations
        error_message = str(e_info.value)
        assert error_message == expected_error, 'Expected exception message {}, retrieved {}'.format(expected_error, error_message)

    @pytest.mark.integrationtest
    def test_when_all_parameters_are_correct_then_returns_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        input_param_dict = {'dummyKey' : 'dummyValue'}
                
        css_data_length = 42
        css_data_location = (4,2)
        css_data_branch_id = 420
        css_data_chainage = 4.2
        css_data = {
            'id' : [test_css_name],
            'length': [css_data_length],
            'xy' : [css_data_location],
            'branchid':[css_data_branch_id],
            'chainage':[css_data_chainage]
        }
        fmd_arg_list = (None, None, None, None, css_data)
        fm_model_data = FMD(fmd_arg_list)

        # 2. Expectations
        return_css = None

        # 3. Run test
        try:
            return_css = runner._generate_cross_section(
                            css_name = test_css_name, 
                            input_param_dict = input_param_dict, 
                            fm_model_data = fm_model_data)
        except Exception as e_info:
            pytest.fail('No expected exception but was risen: {}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_css is not None
        assert return_css.name == test_css_name, 'Expected name {} but was {}'.format( test_css_name, return_css.name)
        assert return_css.length == css_data_length, 'Expected length {} but was {}'.format( css_data_length, return_css.length)
        assert return_css.location == css_data_location, 'Expected location {} but was {}'.format( css_data_location, return_css.location)
        assert return_css.branch == css_data_branch_id, 'Expected branch {} but was {}'.format( css_data_branch_id, return_css.branch)
        assert return_css.chainage == css_data_chainage, 'Expected chainage {} but was {}'.format( css_data_chainage, return_css.chainage)

class Test_get_new_cross_section:
    
    @pytest.mark.unittest
    @pytest.mark.parametrize('css_data', [(None), ({})])
    def test_when_not_given_css_data_then_returns_none(self, css_data):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'

        # 2. Expectations
        return_value = None

        # 3. Run test
        try:
            return_value = runner._get_new_cross_section(
                            name = test_css_name, 
                            input_param_dict = None, 
                            css_data = css_data)
        except Exception as e_info:
            pytest.fail('No expected exception but was risen: {}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is None

    @pytest.mark.unittest
    def test_when_css_data_id_not_found_then_returns_none(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        input_param_dict = {'dummyKey' : 'dummyValue'}
        css_data = {'dummyKey' : 'dummyValue'}

        # 2. Expectations
        return_value = None

        # 3. Run test
        try:
            return_value = runner._get_new_cross_section(
                            name = test_css_name, 
                            input_param_dict = input_param_dict, 
                            css_data = css_data)
        except Exception as e_info:
            pytest.fail('No expected exception but was risen: {}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is None
    
    @pytest.mark.unittest
    def test_when_css_index_not_found_then_returns_none(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        input_param_dict = {'dummyKey' : 'dummyValue'}
        css_data = {'id' : []}

        # 2. Expectations
        return_value = None

        # 3. Run test
        try:
            return_value = runner._get_new_cross_section(
                            name = test_css_name, 
                            input_param_dict = input_param_dict, 
                            css_data = css_data)
        except Exception as e_info:
            pytest.fail('No expected exception but was risen: {}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is None

    @pytest.mark.unittest
    def test_when_css_data_given_but_empty_input_param_dict_then_returns_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        input_param_dict = {'dummyKey' : 'dummyValue'}
        css_data = {'id' : [test_css_name]}

        # 2. Expectations
        return_value = None

        # 3. Run test
        try:
            return_value = runner._get_new_cross_section(
                            name = test_css_name, 
                            input_param_dict = input_param_dict, 
                            css_data = css_data)
        except Exception as e_info:
            pytest.fail('No expected exception but was risen: {}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_value is None

    @pytest.mark.integrationtest
    def test_when_given_css_data_but_no_input_parameters_then_returns_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        input_param_dict = None
        css_data_length = 42
        css_data_location = (4,2)
        css_data_branch_id = 420
        css_data_chainage = 4.2
        css_data = {
            'id' : [test_css_name],
            'length': [css_data_length],
            'xy' : [css_data_location],
            'branchid':[css_data_branch_id],
            'chainage':[css_data_chainage]
        }

        # 2. Expectations
        return_css = None

        # 3. Run test
        try:
            return_css = runner._get_new_cross_section(
                            name = test_css_name, 
                            input_param_dict = input_param_dict, 
                            css_data = css_data)
        except Exception as e_info:
            pytest.fail('No expected exception but was risen: {}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_css is not None
        assert return_css.name == test_css_name, 'Expected name {} but was {}'.format( test_css_name, return_css.name)
        assert return_css.length == css_data_length, 'Expected length {} but was {}'.format( css_data_length, return_css.length)
        assert return_css.location == css_data_location, 'Expected location {} but was {}'.format( css_data_location, return_css.location)
        assert return_css.branch == css_data_branch_id, 'Expected branch {} but was {}'.format( css_data_branch_id, return_css.branch)
        assert return_css.chainage == css_data_chainage, 'Expected chainage {} but was {}'.format( css_data_chainage, return_css.chainage)

    @pytest.mark.integrationtest
    def test_when_given_valid_arguments_then_returns_expected_css(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        test_css_name = 'dummy_css'
        input_param_dict = {'dummyKey' : 'dummyValue'}
        css_data_length = 42
        css_data_location = (4,2)
        css_data_branch_id = 420
        css_data_chainage = 4.2
        css_data = {
            'id' : [test_css_name],
            'length': [css_data_length],
            'xy' : [css_data_location],
            'branchid':[css_data_branch_id],
            'chainage':[css_data_chainage]
        }

        # 2. Expectations
        return_css = None

        # 3. Run test
        try:
            return_css = runner._get_new_cross_section(
                            name = test_css_name, 
                            input_param_dict = input_param_dict, 
                            css_data = css_data)
        except Exception as e_info:
            pytest.fail('No expected exception but was risen: {}'.format(str(e_info)))

        # 4. Verify final expectations
        assert return_css is not None
        assert return_css.name == test_css_name, 'Expected name {} but was {}'.format( test_css_name, return_css.name)
        assert return_css.length == css_data_length, 'Expected length {} but was {}'.format( css_data_length, return_css.length)
        assert return_css.location == css_data_location, 'Expected location {} but was {}'.format( css_data_location, return_css.location)
        assert return_css.branch == css_data_branch_id, 'Expected branch {} but was {}'.format( css_data_branch_id, return_css.branch)
        assert return_css.chainage == css_data_chainage, 'Expected chainage {} but was {}'.format( css_data_chainage, return_css.chainage)

class Test_export_cross_sections:

    @pytest.mark.unittest
    @pytest.mark.parametrize("cross_sections", [(None), ([]), ('') ])
    def test_when_no_cross_sections_then_does_not_raise(self, cross_sections):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        output_dir = 'dummy_dir'

        # 2. run test
        try:
            runner._export_cross_sections(cross_sections, output_dir)
        except Exception as e:
            e_message = str(e)
            pytest.fail('No exception was expected, but given: {}'.format(e_message))

    @pytest.mark.integration
    @pytest.mark.parametrize("output_dir", [(None), ([]), ('') ])
    def test_when_no_output_dir_then_exports_to_default_output_dir(self, output_dir):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        default_output_dir = 'exported_cross_sections'
        input_params = {}
        css_name = 'dummy_name'
        css_length = 0
        css_location = (0,0)
        test_css = CS(input_params, css_name,css_length, css_location)
        cross_sections = [test_css]

        # 2. verify initial expectations
        if os.path.exists(default_output_dir):
            shutil.rmtree(default_output_dir)
    
        # 3. run test
        try:
            runner._export_cross_sections(cross_sections, output_dir)
        except Exception as e:
            e_message = str(e)
            pytest.fail('No exception was expected, but given: {}'.format(e_message))
        
        # 4. Verify final expectations
        assert os.path.exists(default_output_dir)
        assert os.listdir(default_output_dir)
        shutil.rmtree(default_output_dir)

class Test_calculate_css_correction:
    
    @pytest.mark.unittest
    def test_when_cross_section_not_given_then_no_exception_risen(self):
        pytest.fail('To Do')

    @pytest.mark.integrationtest
    def test_when_all_parameters_are_correct_then_calculates_css_correction(self):
        # 1. Set up test data
        runner = Fm2ProfRunner(None)
        input_param_dict = {
            'sdstorage':'1',

        }
        css_name = 'dummy_name'
        css_length = 0
        css_location = (0,0)
        test_css = CS(input_param_dict, css_name, css_length, css_location )
        starttime = datetime.datetime.now()

        # 2. Set up / verify initial expectations
        assert runner is not None
        assert test_css is not None
        assert test_css._css_is_corrected == False
        
        # 3. Run test
        try:
            runner._calculate_css_correction(input_param_dict,test_css,starttime)
        except:
            pytest.fail('Unexpected exception while calculating css correction.')

        # 4. Verify final expectations.
        assert test_css._css_is_corrected == True
        pytest.fail('Test still needs work.')

class Test_reduce_css_points:

    @pytest.mark.integrationtest
    def test_when_all_parameters_are_correct_then_reduce_points(self):
        # set up test data
        new_number_of_css_points = 25
        old_number_of_css_points = 30

        runner = Fm2ProfRunner(None)
        input_param_dict = {
            'number_of_css_points': str(new_number_of_css_points)
        }
        
        css_name = 'dummy_name'
        css_length = 0
        css_location = (0,0)
        test_css = CS(input_param_dict, css_name, css_length, css_location )
        starttime = datetime.datetime.now()

        # initial expectation
        assert runner is not None
        assert test_css is not None
        assert test_css._css_is_reduced is False
        test_css._css_total_width = [0] * old_number_of_css_points
        test_css._css_z = [1] * old_number_of_css_points
        test_css._css_flow_width = [1] * old_number_of_css_points

        # run
        try:
            runner._reduce_css_points(input_param_dict,test_css,starttime)
        except:
            pytest.fail('Unexpected exception while reducing css points.')

        # verify the final
        assert test_css._css_is_reduced is True
        pytest.fail('Test still needs work.')
