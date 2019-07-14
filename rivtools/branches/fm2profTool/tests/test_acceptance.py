import unittest
import pytest
import sys
import os

import shutil
import matplotlib.pyplot as plt

from tests.TestUtils import TestUtils
import tests.ReportHelper as ReportHelper
from tests.CompareWaalModel import CompareWaalModel as CompareWaalModel
from tests.LatexReport import LatexReport as LatexReport
from tests.HtmlReport import HtmlReport as HtmlReport

from fm2prof.Fm2ProfRunner import Fm2ProfRunner
from fm2prof.Fm2ProfRunner import IniFile

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
    def test_when_output_generated_then_generate_pdf_report(
            self, report_data):

        assert report_data is not None, '' + \
            'No test data available.'

        test_data_dir, cases_and_figures = report_data
        try:
            report = LatexReport()
            latex_report_path = report._generate_python_report(
                test_data_dir, cases_and_figures)
            report._convert_to_pdf(latex_report_path)
        except Exception as e_info:
            err_mssg = 'Error while generating python report ' + \
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

    @classmethod
    def setup_class(Main):
        """
        Sets up the necessary data for Test_Compare_Waal_Model
        """
        TestUtils.install_package('tqdm')

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
            assert os.path.exists(fig_path)


class Test_Compare_Generic_Model:
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

    def __get_geometry_data(self, input_file: str):
        """[summary]

        Arguments:
            input_file {str} -- Geometry csv file

        Returns:
            {tuple} --  Y = chainage; CL = crest level;
                        FPB = floodplain base level;
                        FA = flow area behind summer dike;
                        TA = total area behind summer dike
                        Z = cross-section z values;
                        W = cross-section total width;
                        F = cross-section flow width
        """
        # Reading geometry.csv file
        # Output lists ===

        z_tmp = []
        w_tmp = []
        f_tmp = []
        Y = []
        CL = []
        FPB = []
        FA = []
        TA = []
        n = 0
        assert os.path.exists(input_file), '' + \
            'Input file {} does not exist'.format(input_file)

        with open(input_file) as fin:
            for line in fin:
                ls = line.strip().split(',')
                if 'id' in line[:2]:
                    z_index = ls.index('level')
                    w_index = ls.index('Total width')
                    f_index = ls.index('Flow width')
                    y_index = ls.index('chainage')
                    cl_index = ls.index('Crest level summerdike')
                    fpb_index = ls.index(
                        'Floodplain baselevel behind summerdike')
                    fa_index = ls.index('Flow area behind summerdike')
                    ta_index = ls.index('Total area behind summerdike')
                    sd_key = ls.index('Use Summerdike')
                elif 'meta' ''in line:
                    Y.append(float(ls[y_index]))      # chainage
                    if ls[sd_key] == '1':
                        # crest level
                        CL.append(float(ls[cl_index]))
                        # floodplain base level
                        FPB.append(float(ls[fpb_index]))
                        # flow area behind summer dike
                        FA.append(float(ls[fa_index]))
                        # total area behind summer dike
                        TA.append(float(ls[ta_index]))
                    if n == 1:
                        Z = [z_tmp]
                        W = [w_tmp]
                        F = [f_tmp]
                        n += 1
                    elif n > 1:
                        Z.append(z_tmp)
                        W.append(w_tmp)
                        F.append(f_tmp)
                    z_tmp = []
                    w_tmp = []
                    f_tmp = []
                    n += 1
                elif 'geom' in line:
                    z_tmp.append(float(ls[z_index]))  # z values
                    w_tmp.append(float(ls[w_index]))  # w values (total width)
                    f_tmp.append(float(ls[f_index]))  # fw values (flow width)
            Z.append(z_tmp)
            W.append(w_tmp)
            F.append(f_tmp)

        return (Z, W, F, Y, CL, FPB, FA, TA)

    def __interpolate_z(self, tzw: list, y: float):
        """
        interpolate the analytical cross-section
        at the given chainage y (float)
        """
        tz = []
        tw = []
        y0 = tzw[0][1]
        for i in range(len(tzw)):
            if tzw[i][0] == y:
                tz.append(float(tzw[i][-1]))  # z
                tw.append(float(tzw[i][1]))  # y

            elif tzw[i][0] > y:
                for j in range(i-1, -1, -1):
                    if tzw[j][1] == y0:
                        tzw0 = j
                        tzw1 = i
                        ty0 = tzw[j][0]
                        tz0 = [tzw[z][-1] for z in range(tzw0, tzw1)]
                        tw0 = [tzw[z][1] for z in range(tzw0, tzw1)]
                        ty1 = tzw[i][0]
                        tz1_range = range(tzw1, tzw1 + tzw1 - tzw0)
                        tz1 = [tzw[z][-1] for z in tz1_range]
                        rr = (y-ty0)/(ty1-ty0)
                        tz1_len = range(len(tz1))
                        tz = [(rr*(tz1[x]-tz0[x]))+tz0[x] for x in tz1_len]
                        tw = tw0
                        break
                break
        return tz, tw

    def __Interpolate_tz_for_CS(self, plt_z, plt_x, tz, tx):
        est_z = [plt_z[0]]
        est_x = plt_x
        n = 0
        for i in range(1, len(plt_z) - 1):
            flag1 = 0
            for j in range(n, len(tz)):
                # est_x[i] == float(tx[j]):
                if abs(est_x[i]-float(tx[j])) < 1e-5:
                    est_z.append(tz[j])
                    n = j+1
                    break
                elif est_x[i] < float(tx[j]) and est_x[i] > float(tx[j-1]):
                    est_z.append(tz[j])
                    break
                elif est_x[i] > tx[-1]:
                    est_z.append(plt_z[i])
                    break
                elif abs(est_x[i]-est_x[i+1]) < 1e-5 and \
                        abs(est_x[i] - est_x[i-1]) < 1e-5:
                    est_z.append(tz[-1])
                    flag1 = 1
                    break
            if abs(est_x[i]-est_x[i+1]) < 1e-5 and \
                    abs(est_x[i]-est_x[i-1]) < 1e-5 and flag1 == 0:
                est_z.append(tz[-1])

        est_z.append(plt_z[-1])
        return est_z, est_x

    def __Error_check(self, plt_tz, plt_z, plt_x):
        diff = [plt_tz[i]-plt_z[i] for i in range(len(plt_z))]
        ErrorList = []
        for i in range(len(plt_tz)-1):
            dx = plt_x[i+1]-plt_x[i]
            diff_pos_i = diff[i]
            diff_next_pos = diff[i+1]
            if (diff_pos_i > 0 and diff_next_pos > 0) or \
                    (diff_pos_i < 0 and diff_next_pos < 0):
                min_z = abs(diff_pos_i)
                if abs(diff_pos_i) > abs(diff_next_pos):
                    min_z = abs(diff_next_pos)
                dx = plt_x[i+1]-plt_x[i]
                Error_area = (min_z + 0.5*abs(diff_next_pos-diff_pos_i)) * dx
            elif diff_pos_i == 0 and diff_next_pos == 0:
                Error_area = 0
            elif diff_pos_i == 0 or diff_next_pos == 0:
                dz = abs(diff_pos_i)
                if abs(diff_pos_i) < abs(diff_next_pos):
                    dz = abs(diff_next_pos)
                Error_area = 0.5 * dz * dx
            else:
                D = abs(diff_pos_i/diff_next_pos)
                dx0 = (D*dx)/(D+1)
                dx1 = dx-dx0
                diff_pos_i_dx0 = abs(diff_pos_i) * dx0
                diff_next_pos_dx1 = abs(diff_next_pos) * dx1
                Error_area = 0.5 * (diff_pos_i_dx0 + diff_next_pos_dx1)
            ErrorList.append(Error_area)
        return sum(ErrorList)

    def __ShiftCrossSection(self, tx, tz):
        min_value = min(tz)
        min_list = [i for i, x in enumerate(tz) if x == min_value]
        midpoint = (tx[min_list[-1]]-tx[min_list[0]])/2 + tx[min_list[0]]
        cs_midpoint = (tx[-1]-tx[0])/2 + tx[0]
        shift = midpoint - cs_midpoint
        tmp_tx = [tx[i]-shift for i in range(1, len(tx)-1)]
        new_tx = [tx[0]] + tmp_tx + [tx[-1]]
        return new_tx

    def __symmetric(self, L):
        if len(L) % 2 != 0:
            return False
        else:
            for i in range(1, int(len(L)/2)):
                if L[i]-L[i-1] != L[-i]-L[-i-1] > 1e-4:
                    return False
        return True

    def __plot_cs(
            self, fig_dir: str,
            tx, tz, plt_x, plt_z, y, err,
            plt_s=None, ts=None, ttbs=None, tfpb=None,
            tcl=None, tbs=None, fpb=None, cl=None):
        fig, axh = plt.subplots(1, figsize=(10, 4))
        tz_plt = [plt_z[0]] + tz + [plt_z[-1]]
        tx_plt = [tx[0]] + tx + [tx[-1]]
        if ts is not None and ts:
            ts_plt = [ts[0]] + ts + [ts[-1]]
            axh.plot(
                ts_plt, tz_plt,
                label='Analytical flow width',
                linestyle='--',
                color='#1f77b4')
        if plt_s is not None and plt_s:
            axh.plot(
                plt_s, plt_z,
                label='FM2PROF flow width',
                linestyle=':',
                color='#ff7f0e')
        axh.plot(
            tx_plt, tz_plt,
            label='Analytical total width',
            color='#ff7f0e')
        axh.plot(
            plt_x, plt_z,
            label='FM2PROF total width',
            color='#1f77b4')
        axh.set_ylim()
        axh.set_xlabel('x [m]')
        axh.set_ylabel('z [m]')
        axh.legend()
        titlestr = 'Cross-section at chainage ' + str(y)
        axh.set_title(titlestr)
        axh.text(
            0.06, 0.85, 'sum(err) = {:.2f}'.format(err),
            horizontalalignment='left',
            verticalalignment='center',
            transform=axh.transAxes)
        if ttbs is not None:
            ttbs_text = '' + \
                '\tCrestLevel = {:.2f}m,' + \
                ' Floodplain Base level = {:.2f}m\n' + \
                '\tCrest height = {:.2f}m,' + \
                ' Total area behind summer dike = {:.2f}m^2'
            axh.text(
                0.06, 0.72,
                'FM2PROF:\n' + ttbs_text.format(cl, fpb, cl - fpb, tbs),
                horizontalalignment='left',
                verticalalignment='center',
                transform=axh.transAxes)
            axh.text(
                0.06, 0.52,
                'Analytical:\n' + ttbs_text.format(tcl, tfpb, 1, ttbs),
                horizontalalignment='left',
                verticalalignment='center',
                transform=axh.transAxes)
        plt.grid()
        plt.tight_layout()
        figtitlestr = 'CrossSection_chainage' + str(int(y))
        fig_name = '{}.png'.format(figtitlestr)
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

    def __convert_zw2xz(
            self,
            z: list, w: list, tx: list, max_width: float,
            flow_width=None):
        """ Convert zw lists to xz list for plotting """
        if flow_width:
            w_tmp = [w[i]/2 for i in range(len(w))]
            plt_w_forward = [w_tmp[i]*-1 for i in range(len(w_tmp))]
            plt_w_reverse = w_tmp[::-1]
        else:
            w_tmp = [(max_width-w[i])/2 for i in range(len(w))]
            plt_w_forward = [max_width/2+w[i]/2 for i in range(len(w))]
            plt_w_reverse = w_tmp[::-1]
        plt_w_tmp = plt_w_reverse + plt_w_forward
        plt_w = [plt_w_tmp[i] + tx[0] for i in range(len(plt_w_tmp))]
        plt_z = z[::-1] + z
        return plt_z, plt_w

    def __FlowWidth_check(self, W: list, F: list, case_name: str):
        for i in range(len(W)):
            storage_width = [W[i][j]-F[i][j] for j in range(len(W[i]))]
            sum_storage_width = sum(storage_width)
            if i == 0:
                S = [storage_width]
            else:
                S.append(storage_width)
        return S

    def __interpolate_s(self):
        """ interpolate the analytical storage width """
        ts = [25, 50, 50.000001, 99.999999, 100, 125]
        return ts

    # region for tests
    @pytest.mark.acceptance
    @pytest.mark.requires_output
    @pytest.mark.parametrize(
        'case_name', _test_scenarios_ids, ids=_test_scenarios_ids)
    def test_when_output_exists_then_compare_generic_model_input(
            self, case_name: str):
        if case_name == _waal_case:
            print('This case is tested on another fixture.')
            return
        # 1. Get all necessary output / input directories
        fm2prof_dir = _get_test_case_output_dir(case_name)
        # Data from the above tests is saved directly in fm2prof_dir,
        # not in case_name/output
        # fm2prof_output_dir = os.path.join(fm2prof_dir, 'Output')
        fm2prof_fig_dir = os.path.join(fm2prof_dir, 'Figures')

        geometry_file_name = 'geometry.csv'
        input_geometry_file = os.path.join(fm2prof_dir, geometry_file_name)

        # 2. Verify / create necessary folders and directories
        assert os.path.exists(input_geometry_file), '' + \
            'Input file {} could not be found'.format(input_geometry_file)
        if os.path.exists(fm2prof_fig_dir):
            shutil.rmtree(fm2prof_fig_dir)
        os.makedirs(fm2prof_fig_dir)

        # Read data in geometry.csv
        (Z, W, F, Y, CL, FPB, FA, TA) = self.__get_geometry_data(
            input_geometry_file)

        S = self.__FlowWidth_check(W, F, case_name)

        tzw_values = self.__case_tzw_dict.get(case_name)
        if not tzw_values or tzw_values is None:
            pytest.fail(
                'Test failed, no values retrieved for {}'.format(case_name))
        if Y[-1] > tzw_values[-1][0]:
            pytest.fail(
                'Test failed, redo FM simulation with the maximum chainage\
                     less than or equal to {}'.format(tzw_values[-1][0]))

        # loop over each chainage (cross-section)
        for cs in range(len(Y)):
            y = Y[cs]
            z = Z[cs]
            w = W[cs]
            s = S[cs]
            # give the interpolated analytical result at chainage y
            [tz, tx] = self.__interpolate_z(tzw_values, y)
            # give the interpolated analytical storage width
            # at chainage y for storage_04
            # check whether the analytical cross-section is symmetric
            if not self.__symmetric(tx):
                tx = self.__ShiftCrossSection(tx, tz)
            [plt_z, plt_x] = self.__convert_zw2xz(z, w, tx, max(w))
            [plt_tz, plt_tx] = self.__Interpolate_tz_for_CS(
                plt_z, plt_x, tz, tx)

            plt_s_flag = False
            if sum(s) > 0.001 or case_name == _case04:
                # Storage or false storage width
                plt_s = None
                ts = None
                plt_s_flag = True
                # case04_storage case only
                if Y[cs] >= 1250 and Y[cs] <= 1750 and case_name == _case04:
                    ts = self.__interpolate_s()
                if sum(s) > 0.001:
                    [plt_z, plt_s] = self.__convert_zw2xz(z, s, tx, max(w), 1)
                    plt_s = [plt_s[i]+plt_x[i] for i in range(len(plt_s))]
            if case_name == _case05:
                # Dike case
                # 50m * 1m (crest level-base level) * 2
                # = analytical total area behind summer dike
                ttbs = 2 * 50.0 * 1.0
                # real floodplain is 1m lower than the "floodplain" in the
                # cross-section geometry
                tfpb = tz[1] - 1.0
                # real crest level is the "floodplain"
                # height in the cross-section geometry
                tcl = tz[1]
                # crest level from fm2prof
                cl = CL[cs]
                # floodplain base level from fm2prof
                fpb = FPB[cs]
                # total area behind summer dike from fm2prof
                tbs = TA[cs]

            sumError = self.__Error_check(plt_tz, plt_z, plt_x)
            # dyke

            if plt_s_flag:
                if case_name == _case05:
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError, plt_s, ts, ttbs, tfpb, tcl, tbs, fpb, cl)
                else:
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError, plt_s, ts)
            else:
                if case_name == _case05:
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError, '', '', ttbs, tfpb, tcl, tbs, fpb, cl)
                else:
                    self.__plot_cs(
                        fm2prof_fig_dir, tx, tz, plt_x, plt_z, y,
                        sumError)
