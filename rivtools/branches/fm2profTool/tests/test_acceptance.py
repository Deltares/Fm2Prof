import unittest
import pytest
import sys
import os

import shutil
import matplotlib.pyplot as plt

from tests import TestUtils
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
        'case_02_compound',
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        'case_03_threestage',
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        'case_04_storage',
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        'case_05_dyke',
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        'case_06_plassen',
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        'case_07_triangular',
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz'),
    pytest.param(
        _waal_case,
        'Data\\FM\\FlowFM_fm2prof_map.nc',
        'Data\\cross_section_locations.xyz', marks=pytest.mark.slow)
]


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
    output_dir = _create_test_root_output_dir("RunWithFiles_Output")
    # Create it if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


def _create_test_root_output_dir(dirName=None):
    """
    Create test output directory
    so it's easier to collect output afterwards.
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


class Test_Generate_Latex_Report:

    import fileinput
    def _get_case_figures(self, case_name):
        case_dir = _get_test_case_output_dir(case_name)
        case_fig_dir = os.path.join(case_dir, 'Figures')
        if not os.path.exists(case_fig_dir):
            return []
        return os.listdir(case_fig_dir)

    def _get_all_cases_and_figures(self):
        case_figures = []
        for case in _test_scenarios_ids:
            case_figures.append((case, self._get_case_figures(case)))
        return case_figures

    def _generate_python_section(self, case_values):
        (case_name, case_figures) = case_values
        latex_lines = []
        entry = '\\chapter{{{}}}\t\\label{{sec:{}}}'.format(case_name, case_name)
        latex_lines.append(entry)
        for case_figure in case_figures:
            fig_path = '..\\{}\\Figures\\{}'.format(case_name, case_figure)
            fullpath, fig_label = os.path.split(case_figure)
            fig_template = """
            \\begin{{figure}}[!h]
            \\centering
                \\includegraphics[width=0.95\\textwidth]{{{}}}
                \\caption{{\\small {}}}
                \\label{{fig:{}}}
            \\end{{figure}}""".format(fig_path, fig_label, fig_label)
            latex_lines.append(fig_template)
        return latex_lines

    def _make_pdf(self, report_dir):
        tex_path = 'acceptance_report.tex'
        bibtex = 'acceptance_report'
        log = 'a_r_Log.txt'
        current_wc = os.getcwd()
        os.system('cd {}'.format(report_dir))
        try:
            os.system('pdflatex {}'.format(tex_path))
            os.system('bibtex {}'.format(bibtex))
            os.system('pdflatex {}'.format(tex_path))
            os.system('pdflatex \'{}\' > {}'.format(tex_path, log))
            os.system('xcopy *.pdf .. /Y')
        except Exception as e_info:
            print('Error while generating pdf: {}'.format(e_info))
        os.system('cd {}'.format(current_wc))

    @pytest.mark.generate_test_report
    def test_when_output_generated_then_generate_report(self):
        # 1. Find latex template
        latex_key = 'PYTHON_CODE_HERE'
        latex_name = 'acceptance_report.tex'
        latex_dir_name = 'latex_report'
        pdf_name = 'acceptance_report.pdf'
        latex_dir = TestUtils.get_test_data_dir(latex_dir_name)
        output_dir = _get_base_output_dir()
        report_dir = os.path.join(output_dir, latex_dir_name)

        # 2. Gather case figures.
        cases_and_figures = self._get_all_cases_and_figures()
        lines = []
        for case in cases_and_figures:
            lines.append(self._generate_python_section(case))

        assert lines

        # 3. Copy dir to output test (to use it later as external)
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)
        shutil.copytree(latex_dir, report_dir, False, None)
        latex_path = os.path.join(report_dir, latex_name)

        assert os.path.exists(latex_path), 'Latex template could not be found on path {}.'.format(latex_path)

        # 4. Add templates to the latex file
        with open(latex_path) as f:
            file_str = f.read()

        new_sections = ['\n'.join(line) for line in lines]
        new_content = '\n\n'.join(new_sections)
        # file_str = file_str.replace(latex_key, new_content)
        file_str = file_str.replace(latex_key, 'test')

        with open(latex_path, 'w') as f:
            f.write(file_str)
        
        # 5. Execute Pdf generator
        self._make_pdf(report_dir)

        # 6. Verify PDF is generated
        pdf_path = os.path.join(report_dir, pdf_name)
        assert os.path.exists(pdf_path), 'PDF file was not generated.'


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

    @pytest.mark.systemtest
    def test_when_given_inifile_then_output_is_generated(self):
        # 1. Set up test data.
        case_name = 'case_01_rectangle'
        map_file = 'Data\\FM\\FlowFM_fm2prof_map.nc'
        css_file = 'Data\\cross_section_locations.xyz'
        root_output_dir = os.path.join(
            os.path.dirname(__file__), "RunMainWithCustomIniFile", case_name)
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
        assert generated_files, "There is no output generated for {0}".format(case_name)
        for expected_file in expected_files:
            assert expected_file in generated_files


class Test_Acceptance_Waal:
    """Requires fm2prof output generated for waal_case
    """

    def __copy_output_to_sobek_dir(self, output_dir: str, target_dir: str):
        """Moves the output generated in output_dir to
        the pre-defined sobek directory.

        Arguments:
            output_dir {str} -- Path location for the directory.
            target_dir {str} -- Path location for the directory.
        """
        if not os.path.exists(output_dir):
            pytest.fail('Output directory {} not found.'.format(output_dir))

        shutil.move(output_dir, target_dir)

    def __create_xml_waal(self, working_dir: str):
        # write file
        file_path = os.path.join(working_dir, 'waal.xml')
        if os.path.exists(file_path):
            os.remove(file_path)
        flow1d_working_dir = os.path.join(working_dir, 'dflow1d')
        f = open(file_path, 'w+')
        f.write('<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n')
        f.write(
            '<dimrConfig xmlns="http://schemas.deltares.nl/dimr"' +
            ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' +
            ' xsi:schemaLocation="http://schemas.deltares.nl/dimr' +
            ' http://content.oss.deltares.nl/schemas/dimr-1.2.xsd">\n')
        f.write('<documentation>\n')
        f.write('<fileVersion>1.2</fileVersion>\n')
        f.write('<createdBy>Deltares, Coupling Team</createdBy>\n')
        f.write('<creationDate>2019-03-28T08:04:13.0106499Z</creationDate>\n')
        f.write('</documentation>\n')
        f.write('<control>\n')
        f.write('<start name="rijn-flow-model" />\n')
        f.write('</control>\n')
        f.write('<component name="rijn-flow-model">\n')
        f.write('<library>cf_dll</library>\n')
        f.write('<workingDir>{}</workingDir>\n'.format(flow1d_working_dir))
        f.write('<inputFile>rijn-flow-model.md1d</inputFile>\n')
        f.write('</component>\n')
        f.write('</dimrConfig>\n')
        f.close()
        return file_path

    def __run_dimr_from_command(self, sobek_xml_location: str):
        """ Runs created xml with dimr script.

        Arguments:
            sobek_xml_location {str}
                -- Location of xml file that points to the working dir.
        """
        dimr_runner_relative = ('SOBEK\\plugins\\DeltaShell.Dimr'
                                '\\kernels\\x64\\dimr\\scripts\\run_dimr.bat')
        dimr_runner_path = os.path.join(
            '..\\waal_sobek_runner',
            dimr_runner_relative)

        dimr_call = '{} {} -d 0 > out.txt 2>&1'.format(
                    dimr_runner_path, sobek_xml_location)
        try:
            os.system(dimr_call)
        except Exception as e_error:
            pytest.fail(
                'Exception thrown while not expected. {}'.format(
                    str(e_error)))

    def __get_observations_file(self, sobek_dir: str):
        """Finds the observations.nc file generated from the sobek dimr run.

        Arguments:
            sobek_dir {str} -- Parent directory used for running dimr

        Returns:
            {str} -- File path where the observation file has been generated.
        """
        dflow1_output = os.path.join(sobek_dir, 'dflow1d\\output')
        assert os.path.exists(dflow1_output), 'Sobek output folder not created.'
        assert os.listdir(dflow1_output), 'No output generated.'

        observations_file = os.path.join(dflow1_output, 'observations.nc')
        assert os.path.exists(observations_file)
        return observations_file

    def __compare_1d_2d_output_and_generate_plots(
            self, case_name: str,
            output_1d: str, output_2d: str,
            fig_dir: str):
        """Compares two .nc files and outputs its result as plots

        Arguments:
            case_name {str} -- Name of the current study case.
            output_1d {str} -- location of the 1d output directory.
            output_2d {str} -- Location of the 2d output directory.
            fig_dir {str} -- Directory where to save the figures.

        Returns:
            {list[str]} -- List of generated figures.
        """

        # Imports
        import numpy as np
        import pandas as pd
        from netCDF4 import Dataset
        import matplotlib
        from tqdm import tqdm
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        font = {'family': 'sans-serif',
                'sans-serif': ['Sansa Pro, sans-serif'],
                'weight': 'normal',
                'size': 20}

        matplotlib.rc('font', **font)
        list_of_figures = []
        # Read data
        df_1d = Dataset(output_1d)
        df_2d = Dataset(output_2d)
        econding = 'utf-8'

        # Parse station names
        stations_1d = np.array(
            ["".join(
                [i.decode(econding).strip() for i in row]
                ) for row in df_1d.variables['observation_id'][:]])
        stations_2d = np.array(
            ["".join(
                [i.decode(econding) for i in row.compressed()]
                ) for row in df_2d.variables['station_name'][:]])
        qstations_2d = np.array(
            ["".join(
                [i.decode(econding) for i in row.compressed()]
                ) for row in df_2d.variables['cross_section_name'][:]])

        # Parse time (to days)
        time_key = 'time'
        t_1d = df_1d.variables[time_key][:]/3600/24
        t_2d = df_2d.variables[time_key][:]/3600/24

        # times at which to compare
        tbnd = [np.max((t_1d[0], t_2d[0])), np.min((t_1d[-1], t_2d[-2]))]
        tinterp = np.linspace(tbnd[0], tbnd[1], 200)

        # calculate bias/std at riverkms
        bias = []
        std = []
        kms = np.arange(868, 961, 1)
        # kms = np.arange(880, 881)
        plot_at = [880, 914, 930, 940, 950, 960]
        plot_at = [960]

        # Keys
        key_1d_water_level = 'water_level'
        key_1d_water_disch = 'water_discharge'
        key_2d_water_level = 'waterlevel'
        key_2d_water_disch = 'cross_section_discharge'

        for km in tqdm(kms):
            stat = '{}.00_WA'.format(km)

            # Find corresponding station for both models
            id_1d = np.argwhere(stations_1d == stat)[0]
            wl_1d = df_1d.variables[key_1d_water_level][:][:, id_1d].flatten()
            q_1d = df_1d.variables[key_1d_water_disch][:][:, id_1d].flatten()

            id_2d = np.argwhere(stations_2d == stat)[0]
            wl_2d = df_2d.variables[key_2d_water_level][:, id_2d].flatten()
            q_2d = df_2d.variables[key_2d_water_disch][id_2d].flatten()

            # compare the two
            interp1d = np.interp(tinterp, t_1d, wl_1d)
            interp2d = np.interp(tinterp, t_2d, wl_2d)
            diffd = interp1d-interp2d

            # append to lists
            bias.append(np.mean(diffd))
            std.append(np.std(diffd))

            # If plot, plot
            if km in plot_at:
                fig, axh = plt.subplots(1, figsize=(10, 4))

                axh.plot(t_1d, wl_1d, label='SOBEK')
                axh.plot(t_2d, wl_2d, label='FM-2D')

                axh.set_ylim()
                axh.set_xlabel('Tijd [dagen]')
                axh.set_ylabel('Waterstand [m + NAP]')
                axh.legend()
                axh.set_title(stat)
                plt.tight_layout()
                fig_name = os.path.join(
                    fig_dir,
                    'case8_{}.png'.format(stat))
                fig.savefig(fig_name)
                list_of_figures.append(fig_name)

        # Plot bias/std
        fig, ax = plt.subplots(1, figsize=(10, 4))
        ax.plot(kms, bias, label='bias')
        ax.plot(kms, std, label='$\\sigma$')
        ax.plot([kms[0], kms[-1]], [0, 0], '--k')
        # ax.plot([913.5]*2, [0, 0.75], '-r')
        ax.legend()
        ax.set_xlabel("Rivierkilometer")
        ax.set_ylabel("Bias/$\\sigma$ [m]")
        ax.set_ylim([-0.25, 1])
        ax.set_xlim([kms[0], kms[-1]])

        plt.tight_layout()
        fig_name = os.path.join(
            fig_dir,
            '{}_statistics.png'.format(case_name))
        fig.savefig(fig_name)
        list_of_figures.append(fig_name)

        # Plot Q/H at selected stations
        stations = [['Q-TielWaal', "LMW.TielWaal", "TielWaal"],
                    ['Q-Nijmegenhaven', 'LMW.Nijmegenhave', "Nijmegenhaven"],
                    ['Q-Zaltbommel', 'LMW.Zaltbommel', "Zaltbommel_waq"],
                    ['Q-Vuren', 'LMW.Vuren', "Vuren"]]

        for station in stations:
            id_1d = np.argwhere(stations_1d == station[1])[0]
            wl_1d = df_1d.variables[key_1d_water_level][:][:, id_1d].flatten()
            q_1d = df_1d.variables[key_1d_water_disch][:][:, id_1d].flatten()

            id_2d = np.argwhere(stations_2d == station[2])[0]
            qid_2d = np.argwhere(qstations_2d == station[0])[0]
            wl_2d = df_2d.variables[key_2d_water_level][:, id_2d].flatten()
            q_2d = df_2d.variables[key_2d_water_disch][:, qid_2d].flatten()

            fig, ax = plt.subplots(1)
            ax.plot(q_1d, wl_1d, '.')
            ax.plot(q_2d, wl_2d, '+')
            ax.set_xlabel('Afvoer [m$^3$/s]')
            ax.set_ylabel('Waterstand [m + NAP]')
            ax.set_title(station[2])

            fig, ax = plt.subplots(1)
            ax.plot(t_1d, q_1d, label='sobek')
            ax.plot(t_2d, q_2d, label='FM2D')

        # fig_path = os.path.join(fig_dir, '{}.png'.format(case_name))
        # plt.savefig(fig_path)
        return list_of_figures

    @pytest.mark.waal_compare_results
    def test_when_results_available_then_compare(self):
        # 1. Set up test data
        waal_test_folder = TestUtils.get_external_test_data_dir(_waal_case)
        sobek_dir = os.path.join(waal_test_folder, 'Model_SOBEK')
        fm_dir = os.path.join(waal_test_folder, 'Model_FM')
        fm2prof_dir = _get_test_case_output_dir(_waal_case)
        compare_dir = os.path.join(fm2prof_dir, 'NC_Output')
        figure_dir = os.path.join(fm2prof_dir, 'Figures')

        if os.path.exists(figure_dir):
            shutil.rmtree(figure_dir)
        os.makedirs(figure_dir)
        # 4. Get observations.nc
        output_1d = self.__get_observations_file(sobek_dir)
        output_2d = os.path.join(fm_dir, 'resultaten\\FlowFM_his.nc')
        assert os.path.exists(output_2d)

        if os.path.exists(compare_dir):
            shutil.rmtree(compare_dir)
        os.makedirs(compare_dir)
        shutil.copy(output_1d, compare_dir)
        shutil.copy(output_2d, compare_dir)
        output_1d = os.path.join(compare_dir, 'observations.nc')
        output_2d = os.path.join(compare_dir, 'FlowFM_his.nc')
        assert os.path.exists(output_1d)
        assert os.path.exists(output_2d)

        # 5. Compare values Generate figures.
        figure_list = self.__compare_1d_2d_output_and_generate_plots(
            _waal_case, output_1d, output_2d, figure_dir)

        # 6. Verify final expectations
        assert os.listdir(figure_dir)
        assert figure_list
        for fig_path in figure_list:
            assert os.path.exists(fig_path)

    @pytest.mark.slow
    @pytest.mark.acceptance
    @pytest.mark.requires_output
    def test_when_output_exists_then_use_it_for_sobek_model_input(self):
        # 1. Set up test data
        waal_test_folder = TestUtils.get_external_test_data_dir(_waal_case)
        sobek_dir = os.path.join(waal_test_folder, 'Model_SOBEK')
        fm_dir = os.path.join(waal_test_folder, 'Model_FM')
        fm2prof_dir = _get_test_case_output_dir(_waal_case)
        compare_dir = os.path.join(fm2prof_dir, 'NC_Output')
        figure_dir = os.path.join(fm2prof_dir, 'Figures')

        # 2. Verify existent output dir
        if not os.path.exists(fm2prof_dir):
            pytest.fail(
                'Fm2Prof output dir {} not found.'.format(fm2prof_dir))
        if not os.path.exists(sobek_dir):
            pytest.fail(
                'Sobek directory not found. {}'.format(sobek_dir))
        if not os.path.exists(fm_dir):
            pytest.fail(
                'FM directory not found. {}'.format(fm_dir))

        if os.path.exists(figure_dir):
            shutil.rmtree(figure_dir)
        os.makedirs(figure_dir)

        # 4. Create xml
        sobek_xml_location = self.__create_xml_waal(sobek_dir)

        # 5. Run DIMR
        self.__run_dimr_from_command(sobek_xml_location)

        # 6. Get observations.nc
        output_1d = self.__get_observations_file(sobek_dir)
        output_2d = os.path.join(fm_dir, 'resultaten\\FlowFM_his.nc')
        assert os.path.exists(output_2d)

        if os.path.exists(compare_dir):
            shutil.rmtree(compare_dir)
        os.makedirs(compare_dir)
        shutil.copy(output_1d, compare_dir)
        shutil.copy(output_2d, compare_dir)
        output_1d = os.path.join(compare_dir, 'observations.nc')
        output_2d = os.path.join(compare_dir, 'FlowFM_his.nc')
        assert os.path.exists(output_1d)
        assert os.path.exists(output_2d)

        # 7. Compare values Generate figures.
        figure_list = self.__compare_1d_2d_output_and_generate_plots(
            _waal_case, output_1d, output_2d, fm2prof_dir)

        # 8. Verify final expectations
        assert figure_list
        for fig_path in figure_list:
            assert os.path.exists(fig_path)


class Test_Acceptance_Generic:
    # region of helpers
    __case_01_tzw = [
        [0,0,0],
        [0,150,0],
        [3000,0,-1],
        [3000,150,-1]]

    __case_02_tzw = [
        [0,0,0],
        [0,50,0],
        [0,50.000001,-2],
        [0,99.999999,-2],
        [0,100,0],
        [0,150,0],
        [3000,0,-1],
        [3000,50,-1],
        [3000,50.000001,-3],
        [3000,99.999999,-3],
        [3000,100,-1],
        [3000,150,-1]]

    __case_03_tzw = [
        [0,0,0.5],
        [0,25,0.5],
        [0,25.000001,0],
        [0,50,0],
        [0,50.000001,-2],
        [0,99.999999,-2],
        [0,100,0],
        [0,124.999999,0],
        [0,125,0.5],
        [0,150,0.5],
        [3000,0,-0.5],
        [3000,25,-0.5],
        [3000,25.000001,-1],
        [3000,50,-1],
        [3000,50.000001,-3],
        [3000,99.999999,-3],
        [3000,100,-1],
        [3000,124.999999,-1],
        [3000,125,-0.5],
        [3000,150,-0.5]]

    __case_05_tzw = [
        [0,0,1],
        [0,50,1],
        [0,50.000001,-2],
        [0,99.999999,-2],
        [0,100,1],
        [0,150,1],
        [3000,0,0],
        [3000,50,0],
        [3000,50.000001,-3],
        [3000,99.999999,-3],
        [3000,100,0],
        [3000,150,0]]

    __case_07_tzw = [
        [0,6000,0],
        [0,6125,0], 
        [0,6125.000001,-2],
        [0,6374.999999,-2],
        [0,6375,0],
        [0,6500,0],
        [250,6000,-0.1],
        [250,6226,-0.1], 
        [250,6226.000001,-2.1],
        [250,6407.999999,-2.1],
        [250,6408,-0.1],
        [250,6500,-0.1],
        [500,6000,-0.2],
        [500,6200,-0.2], 
        [500,6200.000001,-2.2],
        [500,6399.999999,-2.2],
        [500,6400,-0.2],
        [500,6500,-0.2],
        [9500,6000,-1.8],
        [9500,6200,-1.8], 
        [9500,6200.000001,-3.8],
        [9500,6399.999999,-3.8],
        [9500,6400,-1.8],
        [9500,6500,-1.8],
        [9750,6000,-1.9],
        [9750,6182,-1.9], 
        [9750,6182.000001,-3.9],
        [9750,6391.999999,-3.9],
        [9750,6392,-1.9],
        [9750,6500,-1.9],
        [10000,6000,-2],
        [10000,6125,-2],
        [10000,6125.000001,-4],
        [10000,6374.999999,-4],
        [10000,6375,-2],
        [10000,6500,-2]]

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
        Y=[]
        CL = []
        FPB = []
        FA = []
        TA = []
        n = 0
        assert os.path.exists(input_file), 'Input file {} does not exist'.format(input_file)
        
        with open(input_file) as fin:
            for line in fin:
                ls = line.strip().split(',')
                if 'id' in line[:2]:
                    z_index = ls.index('level')
                    w_index = ls.index('Total width')
                    f_index = ls.index('Flow width')
                    y_index = ls.index('chainage')
                    cl_index = ls.index('Crest level summerdike')
                    fpb_index = ls.index('Floodplain baselevel behind summerdike')
                    fa_index = ls.index('Flow area behind summerdike')
                    ta_index = ls.index('Total area behind summerdike')
                    sd_key = ls.index('Use Summerdike')
                elif 'meta' ''in line:
                    Y.append(float(ls[y_index]))      # chainage
                    if ls[sd_key] == '1':
                        CL.append(float(ls[cl_index]))    # crest level
                        FPB.append(float(ls[fpb_index]))  # floodplain base level
                        FA.append(float(ls[fa_index]))    # flow area behind summer dike
                        TA.append(float(ls[ta_index]))    # total area behind summer dike
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
        interpolate the analytical cross-section at the given chainage y (float)
        """
        tz=[]
        tw=[]
        y0 = tzw[0][1]
        for i in range(len(tzw)):
            if tzw[i][0] == y:
                tz.append(float(tzw[i][-1])) # z
                tw.append(float(tzw[i][1])) # y
                
            elif tzw[i][0] > y:
                for j in range(i-1,-1,-1):
                    if tzw[j][1] == y0:
                        tzw0 = j
                        tzw1 = i
                        ty0 = tzw[j][0]
                        tz0 = [tzw[z][-1] for z in range(tzw0,tzw1)]
                        tw0 = [tzw[z][1] for z in range(tzw0,tzw1)]
                        ty1 = tzw[i][0]
                        tz1 = [tzw[z][-1] for z in range(tzw1,tzw1+tzw1-tzw0)]
                        rr = (y-ty0)/(ty1-ty0)
                        tz = [(rr*(tz1[x]-tz0[x]))+tz0[x] for x in range(len(tz1))]
                        tw = tw0
                        break
                break
        return tz,tw

    def __Interpolate_tz_for_CS(self, plt_z, plt_x, tz, tx):
        est_z = [plt_z[0]]
        est_x = plt_x
        n = 0
        outputlist = []
        for i in range(1,len(plt_z)-1):
            flag1 = 0
            for j in range(n,len(tz)):
                if abs(est_x[i]-float(tx[j]))<1e-5:# est_x[i] == float(tx[j]):
                    est_z.append(tz[j])
                    n = j+1
                    outputlist.append([plt_x[i],tz[j],plt_z[i],i,j,n,'I'])
                    break
                elif est_x[i] < float(tx[j]) and est_x[i] > float(tx[j-1]):
                    est_z.append(tz[j])
                    outputlist.append([plt_x[i],tz[j],plt_z[i],i,j,n,'III'])
                    break
                elif est_x[i] > tx[-1]:
                    est_z.append(plt_z[i])
                    break
                elif abs(est_x[i]-est_x[i+1])<1e-5 and abs(est_x[i]-est_x[i-1])<1e-5:
                    est_z.append(tz[-1])
                    flag1 = 1
                    break
            if abs(est_x[i]-est_x[i+1])<1e-5 and abs(est_x[i]-est_x[i-1])<1e-5 and flag1 == 0:
                est_z.append(tz[-1])
            
        est_z.append(plt_z[-1])
        return est_z, est_x, outputlist

    def __Error_check(self, plt_tz, plt_z, plt_x):
        diff = [plt_tz[i]-plt_z[i] for i in range(len(plt_z))]
        ErrorList = []
        for i in range(len(plt_tz)-1):
            dx = plt_x[i+1]-plt_x[i]
            if (diff[i] > 0 and diff[i+1] > 0) or (diff[i] < 0 and diff[i+1] < 0):
                min_z = abs(diff[i])
                if abs(diff[i]) > abs(diff[i+1]):
                    min_z = abs(diff[i+1])
                dx = plt_x[i+1]-plt_x[i]
                Error_area = (min_z + 0.5*abs(diff[i+1]-diff[i])) *dx
            elif diff[i] == 0 and diff[i+1] == 0:
                Error_area = 0
            elif diff[i] == 0 or diff[i+1] == 0:
                dz = abs(diff[i])
                if abs(diff[i]) < abs(diff[i+1]):
                    dz = abs(diff[i+1])
                Error_area = 0.5* dz * dx
            else:
                D = abs(diff[i]/diff[i+1])
                dx0 = (D*dx)/(D+1)
                dx1 = dx-dx0
                Error_area = 0.5 * (abs(diff[i])*dx0 + abs(diff[i+1])*dx1 )
            ErrorList.append(Error_area)
        return sum(ErrorList)

    def __ShiftCrossSection(self,tx,tz):
        min_value = min(tz)
        min_list = [i for i, x in enumerate(tz) if x == min_value]
        midpoint = (tx[min_list[-1]]-tx[min_list[0]])/2 + tx[min_list[0]]
        cs_midpoint = (tx[-1]-tx[0])/2 + tx[0]
        shift = midpoint - cs_midpoint
        tmp_tx = [tx[i]-shift for i in range(1,len(tx)-1)]
        new_tx = [tx[0]] + tmp_tx + [tx[-1]]
        return new_tx

    def __symmetric(self,L):
        if len(L)%2 != 0:
            return False
        else:
            for i in range(1,int(len(L)/2)):
                if L[i]-L[i-1] != L[-i]-L[-i-1] > 1e-4:
                    return False
        return True

    def __plot_cs(self, fig_dir: str, tx,tz,plt_x,plt_z,y,err,ttbs=None,tfpb=None,tcl=None,tbs=None,fpb=None,cl=None):
        fig, axh = plt.subplots(1, figsize=(10, 4))
        axh.plot(tx, tz, label='Analytical')
        axh.plot(plt_x, plt_z, label='FM2PROF')
        axh.set_ylim()
        axh.set_xlabel('x [m]')
        axh.set_ylabel('z [m]')
        axh.legend()
        titlestr = 'Cross-section at chainage ' + str(y)
        axh.set_title(titlestr)
        axh.text(0.06, 0.85, 'sum(err) = {:.2f}'.format(err),
        horizontalalignment='left',
        verticalalignment='center',
        transform = axh.transAxes)
        if ttbs is not None:
            axh.text(0.06, 0.72, 'FM2PROF:\n    CrestLevel = {:.2f}m, Floodplain Base level = {:.2f}m\n    Crest height = {:.2f}m, Total area behind summer dike = {:.2f}m^2'.format(cl,fpb,cl-fpb,tbs),
            horizontalalignment='left',
            verticalalignment='center',
            transform = axh.transAxes)
            axh.text(0.06, 0.52, 'Analytical:\n    CrestLevel = {:.2f}m, Floodplain Base level = {:.2f}m\n    Crest height = {:.2f}m, Total area behind summer dike = {:.2f}m^2'.format(tcl,tfpb,1,ttbs),
            horizontalalignment='left',
            verticalalignment='center',
            transform = axh.transAxes)
        plt.grid()
        plt.tight_layout()
        figtitlestr = 'CrossSection_chainage' + str(int(y))
        fig_name = '{}.png'.format(figtitlestr)
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)

    def __convert_zw2xz(self, z:list, w:list, tx:list):
        """ Convert zw lists to xz list for plotting """
        w_tmp = [(w[-1]-w[i])/2 for i in range(len(w))]
        plt_w_reverse = w_tmp[::-1]
        plt_w_forward = [w[i]/2+w[-1]/2 for i in range(len(w))]
        plt_w_tmp = plt_w_reverse + plt_w_forward
        plt_w = [plt_w_tmp[i] + tx[0] for i in range(len(plt_w_tmp))]
        plt_z = z[::-1] + z
        return plt_z, plt_w

    def __FlowWidth_check(self, W:list, F:list, case_name:str):
        for i in range(len(W)):
            storage_width = [W[i][j]-F[i][j] for j in range(len(W[i]))]
            sum_storage_width = sum(storage_width)
            if sum_storage_width > 0.001 and case_name is not _case04:
                pytest.fail('Storage width is found!')

    # region for tests
    @pytest.mark.acceptance
    @pytest.mark.requires_output
    @pytest.mark.parametrize(
        'case_name', _test_scenarios_ids, ids=_test_scenarios_ids)
    def test_when_output_exists_then_compare_generic_model_input(self, case_name: str):
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
        assert os.path.exists(input_geometry_file), 'Input file {} could not be found'.format(input_geometry_file)
        if os.path.exists(fm2prof_fig_dir):
            shutil.rmtree(fm2prof_fig_dir)
        os.makedirs(fm2prof_fig_dir)

        # Read data in geometry.csv
        (Z, W, F, Y, CL, FPB, FA, TA) = self.__get_geometry_data(input_geometry_file)
        
        self.__FlowWidth_check(W, F, case_name)

        tzw_values = self.__case_tzw_dict.get(case_name)
        if not tzw_values or tzw_values is None:
            pytest.fail('Test failed, no values retrieved for {}'.format(case_name))
        if Y[-1] > tzw_values[-1][0]:
            pytest.fail('Test failed, redo FM simulation with the maximum chainage less than equal to {}'.format(tzw_values[-1][0]))

        # loop over each chainage (cross-section)
        for cs in range(len(Y)):
            y = Y[cs]
            z = Z[cs]
            w = W[cs]
            # give the interpolated analytical result at chainage y
            [tz, tx] = self.__interpolate_z(tzw_values, y)
            # check whether the analytical cross-section is symmetric
            if not self.__symmetric(tx):
                tx = self.__ShiftCrossSection(tx,tz)
            [plt_z,plt_x] = self.__convert_zw2xz(z,w,tx) 
            [plt_tz,plt_tx,outlist] = self.__Interpolate_tz_for_CS(plt_z,plt_x,tz,tx)
            if case_name == _case05:
                # Dike case
                ttbs = 2*50.0*1.0 # 50m * 1m (crest level-base level) * 2 = analytical total area behind summer dike
                tfpb = tz[1] - 1.0 # real floodplain is 1m lower than the "floodplain" in the cross-section geometry
                tcl = tz[1] # real crest level is the "floodplain" height in the cross-section geometry
                cl = CL[cs] # crest level from fm2prof
                fpb = FPB[cs] # floodplain base level from fm2prof
                #fbs = FA[cs] # flow area behind summer dike from fm2prof
                tbs = TA[cs] # total area behind summer dike from fm2prof
            elif case_name == _case04:
                # Storage
                pass         
            elif case_name == _case06:
                # Lake case
                pass                
            
            sumError = self.__Error_check(plt_tz,plt_z,plt_x)
            # dyke
            
            if case_name == _case05:
                self.__plot_cs(fm2prof_fig_dir, tx, tz, plt_x, plt_z, y, sumError, ttbs, tfpb, tcl, tbs, fpb, cl)
            else:
                self.__plot_cs(fm2prof_fig_dir, tx, tz, plt_x, plt_z, y, sumError)
            

