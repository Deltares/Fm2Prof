import unittest
import pytest
import sys
import os

import shutil

from tests import TestUtils
from fm2prof.Fm2ProfRunner import Fm2ProfRunner
from fm2prof.Fm2ProfRunner import IniFile

_root_output_dir = None

# Test data to be used
_waal_case = 'case_08_waal'
_waal_map_file = 'Data\\FM\\FlowFM_fm2prof_map.nc'
_waal_css_file = 'Data\\cross_section_locations.xyz'
_test_scenarios_ids = [
    'case_01_rectangle',
    'case_02_compound',
    'case_03_threestage',
    'case_04_storage',
    'case_05_dyke',
    'case_06_plassen',
    'case_07_triangular',
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
        'case_01_rectangle',
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
        _waal_map_file,
        _waal_css_file, marks=pytest.mark.slow)
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


class Test_Fm2Prof_Run_IniFile:

    @pytest.mark.acceptance
    @pytest.mark.timeout(7200)
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
        assert (os.listdir(iniFile._output_dir),
                "There is no output generated for {0}".format(case_name))


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

    @pytest.mark.system
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
        assert (generated_files,
                "There is no output generated for {0}".format(case_name))
        for expected_file in expected_files:
            assert expected_file in generated_files


class Test_Acceptance_Waal:

    def __run_test(self):
        """Runs the Waal case and returns its output directory location.
        """
        # 1. Set up test data.
        iniFilePath = None
        case_name = _waal_case
        map_file = _waal_map_file
        css_file = _waal_css_file

        iniFile = IniFile.IniFile(iniFilePath)
        test_data_dir = TestUtils.get_external_test_data_dir(_waal_case)
        base_output_dir = _get_base_output_dir()
        iniFile._output_dir = _check_and_create_test_case_output_dir(
            base_output_dir, case_name)
        iniFile._input_file_paths = {
            "fm_netcdfile": os.path.join(test_data_dir, map_file),
            'crosssectionlocationfile': os.path.join(test_data_dir, css_file),
        }

        iniFile._input_parameters = get_valid_inifile_input_parameters()

        # Create the runner and set the saving figures variable to true
        runner = Fm2ProfRunner(iniFilePath)

        # 2. Verify precondition (no output generated)
        assert (os.path.exists(iniFile._output_dir) and
                not os.listdir(iniFile._output_dir))

        # 3. Run file:
        runner.run_inifile(iniFile)

        # 4. Verify there is output generated:
        assert (os.listdir(iniFile._output_dir),
                "There is no output generated for {0}".format(case_name))

        # 5. Return output_dir
        return iniFile._output_dir

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
        f.write('<?xml version="1.0" encoding="utf-8" standalone="yes"?>')
        f.write('<dimrConfig xmlns="http://schemas.deltares.nl/dimr" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schemas.deltares.nl/dimr http://content.oss.deltares.nl/schemas/dimr-1.2.xsd">')
        f.write('<documentation>')
        f.write('<fileVersion>1.2</fileVersion>')
        f.write('<createdBy>Deltares, Coupling Team</createdBy>')
        f.write('<creationDate>2019-03-28T08:04:13.0106499Z</creationDate>')
        f.write('</documentation>')
        f.write('<control>')
        f.write('<start name="rijn-flow-model" />')
        f.write('</control>')
        f.write('<component name="rijn-flow-model">')
        f.write('<library>cf_dll</library>')
        f.write('<workingDir>{}</workingDir>'.format(flow1d_working_dir))
        f.write('<inputFile>rijn-flow-model.md1d</inputFile>')
        f.write('</component>')
        f.write('</dimrConfig>')
        f.close()
        return file_path

    def __run_dimr_from_command(self, sobek_xml_location: str):
        """ Runs created xml with dimr script.

        Arguments:
            sobek_xml_location {str} -- Location of xml file that points to the working dir.
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
        assert (os.path.exists(dflow1_output),
                'Sobek output folder not created.')
        assert os.listdir(dflow1_output), 'No output generated.'

        observations_file = os.path.join(dflow1_output, 'observations.nc')
        assert os.path.exists(observations_file)
        return observations_file

    def __compare_1d_2d_output(self, output_1d, output_2d):
        # Imports
        from fm2prof.main import Fm2ProfRunner
        from fm2prof import utils as futils
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from netCDF4 import Dataset
        import matplotlib
        from tqdm import tqdm

        font = {'family': 'sans-serif',
                'sans-serif': ['Sansa Pro, sans-serif'],
                'weight': 'normal',
                'size': 20}

        matplotlib.rc('font', **font)
        # Read data
        df_1d = Dataset(output_1d)
        df_2d = Dataset(output_2d)

        # Parse station names
        stations_1d = np.array(
            ["".join(
                [i.decode("utf-8").strip() for i in row]
                ) for row in df_1d.variables['observation_id'][:]])
        stations_2d = np.array(
            ["".join(
                [i.decode("utf-8") for i in row.compressed()]
                ) for row in df_2d.variables['station_name'][:]])
        qstations_2d = np.array(
            ["".join(
                [i.decode("utf-8") for i in row.compressed()]
                ) for row in df_2d.variables['cross_section_name'][:]])

        # Parse time (to days)
        t_1d = df_1d.variables['time'][:]/3600/24
        t_2d = df_2d.variables['time'][:]/3600/24

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
        for km in tqdm(kms):
            stat = '{}.00_WA'.format(km)

            # Find corresponding station for both models
            id_1d = np.argwhere(stations_1d == stat)[0]
            wl_1d = df_1d.variables['water_level'][:][:, id_1d].flatten()		
            q_1d = df_1d.variables['water_discharge'][:][:, id_1d].flatten()		

            id_2d = np.argwhere(stations_2d == stat)[0]
            wl_2d = df_2d.variables['waterlevel'][:, id_2d].flatten()
            q_2d = df_2d.variables['cross_section_discharge'][id_2d].flatten()

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
                fig.savefig("case8_{}.png".format(stat))

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
        fig.savefig('case8_statistics.png')

        # Plot Q/H at selected stations
        stations = [['Q-TielWaal', "LMW.TielWaal", "TielWaal"],
                    ['Q-Nijmegenhaven', 'LMW.Nijmegenhave', "Nijmegenhaven"],
                    ['Q-Zaltbommel', 'LMW.Zaltbommel', "Zaltbommel_waq"],
                    ['Q-Vuren', 'LMW.Vuren', "Vuren"]]

        for station in stations:
            id_1d = np.argwhere(stations_1d == station[1])[0]
            wl_1d = df_1d.variables['water_level'][:][:, id_1d].flatten()
            q_1d = df_1d.variables['water_discharge'][:][:, id_1d].flatten()

            id_2d = np.argwhere(stations_2d == station[2])[0]
            qid_2d = np.argwhere(qstations_2d == station[0])[0]
            wl_2d = df_2d.variables['waterlevel'][:, id_2d].flatten()
            q_2d = df_2d.variables['cross_section_discharge'][:, qid_2d].flatten()

            fig, ax = plt.subplots(1)
            ax.plot(q_1d, wl_1d, '.')
            ax.plot(q_2d, wl_2d, '+')
            ax.set_xlabel('Afvoer [m$^3$/s]')
            ax.set_ylabel('Waterstand [m + NAP]')
            ax.set_title(station[2])

            fig, ax = plt.subplots(1)
            ax.plot(t_1d, q_1d, label='sobek')
            ax.plot(t_2d, q_2d, label='FM2D')
        plt.show()

    @pytest.mark.waal_compare_results
    def test_when_results_available_then_compare(self):
        # 1. Set up test data
        waal_test_folder = TestUtils.get_external_test_data_dir(_waal_case)
        sobek_model_dir = os.path.join(waal_test_folder, 'Model_SOBEK')
        fm_model_dir = os.path.join(waal_test_folder, 'Model_FM')

        # 4. Get observations.nc
        output_1d = self.__get_observations_file(sobek_model_dir)
        output_2d = os.path.join(fm_model_dir, 'resultaten\\FlowFM_his.nc')
        assert os.path.exists(output_2d)

        # 5. Compare values Generate figures
        self.__compare_1d_2d_output(output_1d, output_2d)

    @pytest.mark.waal_wihout_running
    def test_when_output_exists_then_use_it_for_sobek_model_input(self):
        # 1. Set up test data
        waal_test_folder = TestUtils.get_external_test_data_dir(_waal_case)
        sobek_model_dir = (waal_test_folder, 'Model_SOBEK')
        fm_model_dir = os.path.join(waal_test_folder, 'Model_FM')

        base_output_dir = _get_base_output_dir()
        fm2prof_output_dir = base_output_dir + "\\{0}".format(_waal_case)

        # 2. Verify existent output dir
        if not os.path.exists(fm2prof_output_dir):
            pytest.fail('Directory {} not found.'.format(fm2prof_output_dir))
        if not os.path.exists(sobek_model_dir):
            pytest.fail(
                'Sobek directory not found. {}'.format(sobek_model_dir))
        if not os.path.exists(fm_model_dir):
            pytest.fail(
                'FM directory not found. {}'.format(fm_model_dir))

        # 4. Create xml
        sobek_xml_location = self.__create_xml_waal(sobek_model_dir)

        # 5. Run DIMR
        self.__run_dimr_from_command(sobek_xml_location)

        # 6. Get observations.nc
        output_1d = self.__get_observations_file(sobek_model_dir)
        output_2d = os.path.join(fm_model_dir, 'resultaten\\FlowFM_his.nc')
        assert os.path.exists(output_2d)

        # 7. Compare values Generate figures
        self.__compare_1d_2d_output(output_1d, output_2d)

    @pytest.mark.waal_running_model
    def test_when_fm2prof_output_then_use_it_for_sobek_model_input(self):
        # 1. Set up test data
        waal_test_folder = TestUtils.get_external_test_data_dir(_waal_case)
        sobek_model_dir = (waal_test_folder, 'Model_SOBEK')
        fm_model_dir = os.path.join(waal_test_folder, 'Model_FM')

        base_output_dir = _get_base_output_dir()
        fm2prof_output_dir = base_output_dir + "\\{0}".format(_waal_case)

        # 2. Run Waal Case through Fm2Prof
        output_dir = self.__run_test()

        # 3. Copy fm2prof output to our own directory
        self.__copy_output_to_sobek_dir(output_dir, fm2prof_output_dir)

        # 4. Create xml
        sobek_xml_location = self.__create_xml_waal(sobek_model_dir)

        # 5. Run DIMR
        self.__run_dimr_from_command(sobek_xml_location)

        # 6. Get observations.nc
        output_1d = self.__get_observations_file(sobek_model_dir)
        output_2d = os.path.join(fm_model_dir, 'resultaten\\FlowFM_his.nc')
        assert os.path.exists(output_2d)

        # 7. Compare values Generate figures
        self.__compare_1d_2d_output(output_1d, output_2d)
