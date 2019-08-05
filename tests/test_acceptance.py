import unittest
import pytest
import sys
import os

import shutil
import matplotlib.pyplot as plt

from tests.TestUtils import TestUtils
from fm2prof.Fm2ProfRunner import Fm2ProfRunner
from fm2prof.Fm2ProfRunner import IniFile

import tests.ReportHelper as ReportHelper
from tests.LatexReport import LatexReport as LatexReport
from tests.HtmlReport import HtmlReport as HtmlReport
from tests.CompareWaalModel import CompareWaalModel as CompareWaalModel
from tests.CompareIdealizedModel \
    import CompareIdealizedModel as CompareIdealizedModel

_root_output_dir = None

# Test data to be used
_waal_case = 'case_08_waal'
_case01 = 'case_01_rectangle'
_case02 = 'case_02_compound'
_case03 = 'case_03_threestage'
_case04 = 'case_04_storage'
_case05 = 'case_05_dyke'
_case06 = 'case_06_plassen'
_case07 = 'case_07_triangular'

_test_scenarios_ids = [
    _case01,
    _case02,
    _case03,
    _case04,
    _case05,
    _case06,
    _case07,
    _waal_case
]

""" To use excluding markups the following command line can be used:
- Include only tests that are acceptance that ARE NOT slow:
    pytest tests -m "acceptance and not slow"
- Include only tests that are both acceptance AND slow:
    pytest tests -m "acceptance and slow"
 """

_test_scenarios = [
    pytest.param(
        _case01,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _case02,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _case03,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _case04,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _case05,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _case06,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _case07,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _waal_case,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz', marks=pytest.mark.slow)
]

_run_with_files_dir_name = 'RunWithFiles_Output'
_base_output_dir_name = 'Output'


def get_valid_inifile_input_parameters():
    return {
        "number_of_css_points": 20,
        "transitionheight_sd": 0.25,
        "velocity_threshold": 0.01,
        "relative_threshold": 0.03,
        "min_depth_storage": 0.02,
        "plassen_timesteps": 10,
        "storagemethod_wli": 1,
        "bedlevelcriterium": 0.1,
        "sdstorage": 1,
        "frictionweighing": 0,
        "sectionsmethod": 0
    }


def _get_base_output_dir():
    """
    Sets up the necessary data for MainMethodTest
    """
    output_dir = _create_test_root_output_dir(_run_with_files_dir_name)
    # Create it if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


def _create_test_root_output_dir(dirName=None):
    """
    Create test output directory
    so it's easier to collect output afterwards.
    """
    _root_output_dir = os.path.join(
        os.path.dirname(__file__), _base_output_dir_name)
    if not os.path.exists(_root_output_dir):
        os.mkdir(_root_output_dir)

    if dirName is not None:
        subOutputDir = os.path.join(_root_output_dir, dirName)
        if not os.path.exists(subOutputDir):
            os.mkdir(subOutputDir)
        return subOutputDir

    return _root_output_dir


def _check_and_create_test_case_output_dir(base_output_dir, caseName):
    """
    Helper to split to set up an output directory
    for the generated data of each test case.
    """
    output_directory = base_output_dir + "\\{0}".format(caseName)

    # clean up the test case output directory if it is no empty
    if os.path.exists(output_directory) and os.listdir(output_directory):
        shutil.rmtree(output_directory)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    return output_directory


def _get_test_case_output_dir(case_name: str):
    base_output_dir = _get_base_output_dir()
    output_directory = base_output_dir + "\\{0}".format(case_name)
    return output_directory


class Test_Generate_Reports:
    _latex_report_path = None

    @pytest.fixture(scope='class')
    def report_data(self):
        """Prepares the class properties to be used in the tests.
        """
        test_data_dir = os.path.join(
            _base_output_dir_name, _run_with_files_dir_name)
        test_data_dir_path = TestUtils.get_test_dir(test_data_dir)
        case_figures = ReportHelper._get_all_cases_and_figures(
            _test_scenarios_ids, test_data_dir_path)
        yield test_data_dir_path, case_figures

    @pytest.mark.generate_test_report
    def test_when_output_generated_then_generate_latex_source_file(
            self, report_data):

        assert report_data is not None, '' + \
            'No test data available.'

        test_data_dir, cases_and_figures = report_data
        try:
            report = LatexReport()
            latex_report_path = report._generate_python_report(
                test_data_dir, cases_and_figures)
        except Exception as e_info:
            err_mssg = 'Error while generating latex source ' + \
                '{}'.format(str(e_info))
            pytest.fail(err_mssg)

    @pytest.mark.generate_test_report
    def test_when_latex_source_generated_then_compile_to_pdf(self):
        
        
        try:
            report = LatexReport()
            report._convert_to_pdf("tests/Output/RunWithFiles_Output/latex_report/")
        except Exception as e_info:
            err_mssg = 'Error while compiling latex report ' + \
                '{}'.format(str(e_info))
            pytest.fail(err_mssg)

    @pytest.mark.generate_test_report
    def test_when_output_generated_then_generate_html_report(
            self, report_data):

        assert report_data is not None, '' + \
            'No test data available.'

        test_data_dir, cases_and_figures = report_data

        try:
            report = HtmlReport()
            report._generate_html_report(
                test_data_dir, cases_and_figures)
        except Exception as e_info:
            err_mssg = 'Error while generating python report ' + \
                '{}'.format(str(e_info))
            pytest.fail(err_mssg)


class Test_Fm2Prof_Run_IniFile:

    @pytest.mark.acceptance
    @pytest.mark.parametrize(
        ("case_name", "map_file", "css_file"),
        _test_scenarios, ids=_test_scenarios_ids)
    def test_when_given_input_data_then_output_is_generated(
            self, case_name, map_file, css_file):
        # 1. Set up test data.
        iniFilePath = None
        iniFile = IniFile.IniFile(iniFilePath)
        test_data_dir = TestUtils.get_external_test_data_dir(case_name)
        base_output_dir = _get_base_output_dir()
        iniFile._output_dir = _check_and_create_test_case_output_dir(
            base_output_dir, case_name)
        iniFile._input_file_paths = {
            "fm_netcdfile": os.path.join(test_data_dir, map_file),
            'crosssectionlocationfile': os.path.join(test_data_dir, css_file),
        }
        iniFile._input_parameters = {
            "number_of_css_points": 20,
            "transitionheight_sd": 0.25,
            "velocity_threshold": 0.01,
            "relative_threshold": 0.03,
            "min_depth_storage": 0.02,
            "plassen_timesteps": 10,
            "storagemethod_wli": 1,
            "bedlevelcriterium": 0.1,
            "sdstorage": 1,
            "frictionweighing": 0,
            "sectionsmethod": 0
        }

        # Create the runner and set the saving figures variable to true
        runner = Fm2ProfRunner(iniFilePath)

        # 2. Verify precondition (no output generated)
        assert (os.path.exists(iniFile._output_dir) and
                not os.listdir(iniFile._output_dir))

        # 3. Run file:
        runner.run_inifile(iniFile)

        # 4. Verify there is output generated:
        assert os.listdir(iniFile._output_dir), '' + \
            'There is no output generated for {0}'.format(case_name)


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

        test_data_dir = TestUtils.get_local_test_data_dir('main_test_data')
        input_file_paths = {
            "fm_netcdfile": os.path.join(test_data_dir, map_file),
            'crosssectionlocationfile': os.path.join(test_data_dir, css_file),
        }
        input_parameters = {
            "number_of_css_points": 20,
            "transitionheight_sd": 0.25,
            "velocity_threshold": 0.01,
            "relative_threshold": 0.03,
            "min_depth_storage": 0.02,
            "plassen_timesteps": 10,
            "storagemethod_wli": 1,
            "bedlevelcriterium": 0.1,
            "sdstorage": 1,
            "frictionweighing": 0,
            "sectionsmethod": 0
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

    def _get_custom_dir(self):
        """
        Sets up the necessary data for MainMethodTest
        """
        output_dir = _create_test_root_output_dir("RunWithCustom_IniFile")
        # Create it if it does not exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        return output_dir

    @pytest.mark.systemtest
    def test_when_given_inifile_then_output_is_generated(self):
        # 1. Set up test data.
        case_name = 'main_case'
        map_file = 'fm_map.nc'
        css_file = 'fm_css.xyz'
        root_output_dir = self._get_custom_dir()
        (ini_file_path, output_dir) = self.__create_test_ini_file(
            root_output_dir, case_name, map_file, css_file)

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
            pytest.fail(
                'No exception expected but was thrown {}.'.format(
                    str(e_error)))

        # 4. Verify there is output generated:
        output_files = os.path.join(output_dir, '{}01'.format(case_name))
        generated_files = os.listdir(output_files)
        if os.path.exists(root_output_dir):
            shutil.rmtree(root_output_dir)
        assert generated_files, '' + \
            'There is no output generated for {0}'.format(case_name)
        for expected_file in expected_files:
            assert expected_file in generated_files


class Test_Compare_Waal_Model:
    """Requires fm2prof output generated for waal_case
    """

    @pytest.mark.slow
    @pytest.mark.acceptance
    @pytest.mark.requires_output
    def test_when_fm2prof_output_then_use_it_for_sobek_model_input(self):
        # 1. Set up test data
        waal_test_folder = TestUtils.get_external_test_data_dir(_waal_case)
        sobek_dir = os.path.join(waal_test_folder, 'Model_SOBEK')
        fm_dir = os.path.join(waal_test_folder, 'Model_FM')
        fm2prof_dir = _get_test_case_output_dir(_waal_case)

        result_figures = []

        # 2. Try to compare.
        try:
            waal_comparer = CompareWaalModel()
            output_1d, _ = waal_comparer._run_waal_1d_model(
                case_name=_waal_case,
                results_dir=fm2prof_dir,
                sobek_dir=sobek_dir,
                fm_dir=fm_dir)
        except Exception as e_info:
            pytest.fail(
                'No exception expected but was thrown ' +
                '{}.'.format(str(e_info)))

        # 3. Verify final expectations
        assert output_1d
        assert os.path.exists(output_1d), '' + \
                'No output found at {}.'.format(output_1d)

    def test_when_sobek_output_exist_then_create_figures(self):
        # 1. Set up test data
        waal_test_folder = TestUtils.get_external_test_data_dir(_waal_case)
        sobek_dir = os.path.join(waal_test_folder, 'Model_SOBEK')
        fm_dir = os.path.join(waal_test_folder, 'Model_FM')
        fm2prof_dir = _get_test_case_output_dir(_waal_case)

        result_figures = []

        # 2. Try to compare.
        try:
            waal_comparer = CompareWaalModel()
            result_figures = waal_comparer._compare_waal(
                case_name=_waal_case,
                results_dir=fm2prof_dir,
                sobek_dir=sobek_dir,
                fm_dir=fm_dir)
        except Exception as e_info:
            pytest.fail(
                'No exception expected but was thrown ' +
                '{}.'.format(str(e_info)))

        # 3. Verify final expectations
        assert result_figures
        for fig_path in result_figures:
            assert os.path.exists(fig_path), '' + \
                'Figure not found at path {}.'.format(fig_path)

    @pytest.mark.acceptance
    @pytest.mark.requires_output
    @pytest.mark.parametrize(
        ("case_name"), _test_scenarios_ids, ids=_test_scenarios_ids)
    def test_when_output_exists_then_compare_waal_model_volume(
            self, case_name: str):
        if case_name != _waal_case:
            # print('This case is tested on another fixture.')
            return
        # 1. Get all necessary output / input directories
        fm2prof_dir = _get_test_case_output_dir(case_name)
        # Data from the above tests is saved directly in fm2prof_dir,
        # not in case_name/output
        fm2prof_fig_dir = os.path.join(fm2prof_dir, 'Figures')

        volume_file_name = 'volumes.csv'
        input_volume_file = os.path.join(fm2prof_dir, volume_file_name)

        # 2. Verify / create necessary folders and directories
        assert os.path.exists(input_volume_file), '' + \
            'Input file {} could not be found'.format(input_volume_file)
        if not os.path.exists(fm2prof_fig_dir):
            os.makedirs(fm2prof_fig_dir)

        #  3. Run
        try:
            waal_comparer = CompareWaalModel()
            waal_comparer._compare_volume(
                    case_name, input_volume_file, fm2prof_fig_dir)
        except Exception as e_info:
            pytest.fail(
                'No exception expected but was thrown ' +
                '{}.'.format(str(e_info)))

        #  4. Final expectation
        assert os.listdir(fm2prof_fig_dir), '' + \
            'There is no volume output generated for {0}'.format(case_name)


class Test_Compare_Idealized_Model:
    # region of helpers
    __case_01_tzw = [
        [0, 0, 0],
        [0, 150, 0],
        [3000, 0, -1],
        [3000, 150, -1]]

    __case_02_tzw = [
        [0, 0, 0],
        [0, 50, 0],
        [0, 50.000001, -2],
        [0, 99.999999, -2],
        [0, 100, 0],
        [0, 150, 0],
        [3000, 0, -1],
        [3000, 50, -1],
        [3000, 50.000001, -3],
        [3000, 99.999999, -3],
        [3000, 100, -1],
        [3000, 150, -1]]

    __case_03_tzw = [
        [0, 0, 0.5],
        [0, 25, 0.5],
        [0, 25.000001, 0],
        [0, 50, 0],
        [0, 50.000001, -2],
        [0, 99.999999, -2],
        [0, 100, 0],
        [0, 124.999999, 0],
        [0, 125, 0.5],
        [0, 150, 0.5],
        [3000, 0, -0.5],
        [3000, 25, -0.5],
        [3000, 25.000001, -1],
        [3000, 50, -1],
        [3000, 50.000001, -3],
        [3000, 99.999999, -3],
        [3000, 100, -1],
        [3000, 124.999999, -1],
        [3000, 125, -0.5],
        [3000, 150, -0.5]]

    __case_05_tzw = [
        [0, 0, 1],
        [0, 50, 1],
        [0, 50.000001, -2],
        [0, 99.999999, -2],
        [0, 100, 1],
        [0, 150, 1],
        [3000, 0, 0],
        [3000, 50, 0],
        [3000, 50.000001, -3],
        [3000, 99.999999, -3],
        [3000, 100, 0],
        [3000, 150, 0]]

    __case_07_tzw = [
        [0, 6000, 0],
        [0, 6125, 0],
        [0, 6125.000001, -2],
        [0, 6374.999999, -2],
        [0, 6375, 0],
        [0, 6500, 0],
        [250, 6000, -0.1],
        [250, 6226, -0.1],
        [250, 6226.000001, -2.1],
        [250, 6407.999999, -2.1],
        [250, 6408, -0.1],
        [250, 6500, -0.1],
        [500, 6000, -0.2],
        [500, 6200, -0.2],
        [500, 6200.000001, -2.2],
        [500, 6399.999999, -2.2],
        [500, 6400, -0.2],
        [500, 6500, -0.2],
        [9500, 6000, -1.8],
        [9500, 6200, -1.8],
        [9500, 6200.000001, -3.8],
        [9500, 6399.999999, -3.8],
        [9500, 6400, -1.8],
        [9500, 6500, -1.8],
        [9750, 6000, -1.9],
        [9750, 6182, -1.9],
        [9750, 6182.000001, -3.9],
        [9750, 6391.999999, -3.9],
        [9750, 6392, -1.9],
        [9750, 6500, -1.9],
        [10000, 6000, -2],
        [10000, 6125, -2],
        [10000, 6125.000001, -4],
        [10000, 6374.999999, -4],
        [10000, 6375, -2],
        [10000, 6500, -2]]

    __case_tzw_dict = {
        _case01: __case_01_tzw,
        _case02: __case_02_tzw,
        _case03: __case_03_tzw,
        _case04: __case_02_tzw,
        _case05: __case_05_tzw,
        _case06: __case_02_tzw,
        _case07: __case_07_tzw,
    }

    # region for tests
    @pytest.mark.acceptance
    @pytest.mark.requires_output
    @pytest.mark.parametrize(
        ("case_name"), _test_scenarios_ids, ids=_test_scenarios_ids)
    def test_when_output_exists_then_compare_generic_model_geometry(
            self, case_name: str):
        if case_name == _waal_case:
            # print('This case is tested on another fixture.')
            return
        # 1. Get all necessary output / input directories
        fm2prof_dir = _get_test_case_output_dir(case_name)
        # Data from the above tests is saved directly in fm2prof_dir,
        # not in case_name/output
        fm2prof_fig_dir_head = os.path.join(fm2prof_dir, 'Figures')
        fm2prof_fig_dir = os.path.join(fm2prof_fig_dir_head, 'Geometry')

        geometry_file_name = 'geometry.csv'
        input_geometry_file = os.path.join(fm2prof_dir, geometry_file_name)

        # 2. Verify / create necessary folders and directories
        assert os.path.exists(input_geometry_file), '' + \
            'Input file {} could not be found'.format(input_geometry_file)
        
        if os.path.exists(fm2prof_fig_dir):
            shutil.rmtree(fm2prof_fig_dir)
        os.makedirs(fm2prof_fig_dir)
        #os.makedirs(fm2prof_fig_dir)

        #  3. Run
        tzw_values = self.__case_tzw_dict.get(case_name)
        if not tzw_values or tzw_values is None:
            pytest.fail(
                'Test failed, no values retrieved for {}'.format(case_name))

        try:
            generic_comparer = CompareIdealizedModel()
            generic_comparer._compare_css(
                    case_name, tzw_values,
                    input_geometry_file, fm2prof_fig_dir)
        except Exception as e_info:
            pytest.fail(
                'No exception expected but was thrown ' +
                '{}.'.format(str(e_info)))

        #  4. Final expectation
        assert os.listdir(fm2prof_fig_dir), '' + \
            'There is no geometry output generated for {0}'.format(case_name)

    # region for tests
    @pytest.mark.acceptance
    @pytest.mark.requires_output
    @pytest.mark.parametrize(
        ("case_name"), _test_scenarios_ids, ids=_test_scenarios_ids)
    def test_when_output_exists_then_compare_generic_model_roughness(
            self, case_name: str):
        if case_name == _waal_case:
            # print('This case is tested on another fixture.')
            return
        # 1. Get all necessary output / input directories
        fm2prof_dir = _get_test_case_output_dir(case_name)
        fm2prof_fig_dir = os.path.join(fm2prof_dir, 'Figures', 'Roughness')

        roughness_file_name = 'roughness.csv'
        input_roughness_file = os.path.join(
            fm2prof_dir, roughness_file_name)

        # 2. Verify / create necessary folders and directories
        assert os.path.exists(input_roughness_file), '' + \
            'Input file {} could not be found'.format(input_roughness_file)
        if not os.path.exists(fm2prof_fig_dir):
            os.makedirs(fm2prof_fig_dir)

        #  3. Run
        tzw_values = self.__case_tzw_dict.get(case_name)
        if not tzw_values or tzw_values is None:
            pytest.fail(
                'Test failed, no values retrieved for {}'.format(case_name))

        try:
            generic_comparer = CompareIdealizedModel()
            generic_comparer._compare_roughness(
                    case_name, tzw_values,
                    input_roughness_file, fm2prof_fig_dir)
        except Exception as e_info:
            pytest.fail(
                'No exception expected but was thrown ' +
                '{}.'.format(str(e_info)))

        assert os.listdir(fm2prof_fig_dir), '' + \
            'There is no roughness output generated for {0}'.format(case_name)

    @pytest.mark.acceptance
    @pytest.mark.requires_output
    @pytest.mark.parametrize(
        ("case_name"), _test_scenarios_ids, ids=_test_scenarios_ids)
    def test_when_output_exists_then_compare_generic_model_volume(
            self, case_name: str):
        if case_name == _waal_case:
            # print('This case is tested on another fixture.')
            return
        # 1. Get all necessary output / input directories
        fm2prof_dir = _get_test_case_output_dir(case_name)
        # Data from the above tests is saved directly in fm2prof_dir,
        # not in case_name/output
        fm2prof_fig_dir = os.path.join(fm2prof_dir, 'Figures', 'Volume')

        volume_file_name = 'volumes.csv'
        input_volume_file = os.path.join(fm2prof_dir, volume_file_name)

        # 2. Verify / create necessary folders and directories
        assert os.path.exists(input_volume_file), '' + \
            'Input file {} could not be found'.format(input_volume_file)
        if not os.path.exists(fm2prof_fig_dir):
            os.makedirs(fm2prof_fig_dir)

        #  3. Run
        try:
            generic_comparer = CompareIdealizedModel()
            generic_comparer._compare_volume(
                    case_name, input_volume_file, fm2prof_fig_dir)
        except Exception as e_info:
            pytest.fail(
                'No exception expected but was thrown ' +
                '{}.'.format(str(e_info)))

        #  4. Final expectation
        assert os.listdir(fm2prof_fig_dir), '' + \
            'There is no volume output generated for {0}'.format(case_name)