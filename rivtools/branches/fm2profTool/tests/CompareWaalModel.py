import os
import shutil
import numpy as np
import pandas as pd

from tests.TestUtils import TestUtils

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset


class CompareWaalModel:

    __sobek_runner_dir_name = 'waal_sobek_runner'
    __runner_script = 'dimr\\scripts\\run_dimr.bat'
    __1d_dir_name = 'dflow1d'

    def _compare_waal(
            self,
            case_name: str,
            results_dir: str,
            sobek_dir: str, fm_dir: str):
        """Compares the output of a sobek and fm model run
        based on the results obtained from a fm2prof run
        (in results dir).

        Arguments:
            case_name {str} -- Name of the current case.
            results_dir {str} -- Output results of Fm2Prof.
            sobek_dir {str} -- Directory with Sobek model.
            fm_dir {str} -- Directory with FM model.

        Raises:
            IOError: When results_dir does not exists.
            IOError: When sobek_dir does not exists.
            IOError: When fm_dir does not exists.

        Returns:
            {list} -- List of generated figures.
        """
        # 1. Set up test data
        figure_dir = os.path.join(results_dir, 'Figures')

        # 2. Verify existent output dir
        if not os.path.exists(results_dir):
            raise IOError(
                'Fm2Prof output dir {} not found.'.format(results_dir))
        if not os.path.exists(sobek_dir):
            raise IOError(
                'Sobek directory not found. {}'.format(sobek_dir))
        if not os.path.exists(fm_dir):
            raise IOError(
                'FM directory not found. {}'.format(fm_dir))

        # 3. Clean up output directory of figures.
        if os.path.exists(figure_dir):
            shutil.rmtree(figure_dir)
        os.makedirs(figure_dir)

        # 4. Create xml
        sobek_xml_location = self.__create_xml_waal(case_name, sobek_dir)

        # 5. Run DIMR
        self.__run_dimr_from_command(sobek_xml_location)

        # 6. Get observations.nc
        output_1d, output_2d = self.__get_observations(
            sobek_dir, fm_dir, results_dir
        )

        # 7. Compare values Generate figures.
        figure_list = self.__compare_1d_2d_output_and_generate_plots(
            case_name, output_1d, output_2d, figure_dir)

        return figure_list

    def __get_observations(
            self, sobek_dir: str, fm_dir: str, results_dir: str):
        """Gets the path to observation files for 1d and 2d.

        Arguments:
            sobek_dir {str} -- Directory for sobek model.
            fm_dir {str} -- Directory for FM model.
            results_dir {str} -- Directory where to move the outputs.

        Raises:
            Exception: If 1d output not in results_dir.
            Exception: If 2d output not in results_dir.

        Returns:
            {tuple} -- Tuple with output_1d and output_2d paths.
        """
        # Get outputs.
        output_1d = self.__get_1d_observations_file(sobek_dir)
        output_2d = self.__get_2d_observations_file(fm_dir)

        # Create NC Output folder.
        compare_dir = os.path.join(results_dir, 'NC_Output')
        if os.path.exists(compare_dir):
            shutil.rmtree(compare_dir)
        os.makedirs(compare_dir)
        shutil.copy(output_1d, compare_dir)
        shutil.copy(output_2d, compare_dir)
        output_1d = os.path.join(compare_dir, 'observations.nc')
        output_2d = os.path.join(compare_dir, 'FlowFM_his.nc')

        if not os.path.exists(output_1d):
            raise Exception(
                '1D output was not found at{}.'.format(output_2d))
        if not os.path.exists(output_2d):
            raise Exception(
                '2D output was not found at{}.'.format(output_2d))
        return output_1d, output_2d

    def __copy_output_to_sobek_dir(self, output_dir: str, target_dir: str):
        """Moves the output generated in output_dir to
        the pre-defined sobek directory.

        Arguments:
            output_dir {str} -- Path location for the directory.
            target_dir {str} -- Path location for the directory.
        """
        if not os.path.exists(output_dir):
            raise IOError('Output directory {} not found.'.format(output_dir))

        shutil.move(output_dir, target_dir)

    def __create_xml_waal(self, case_name, working_dir: str):
        """Creats an XML file to be run by DIMR.

        Arguments:
            case_name {str} -- Name for the current case.
            working_dir {str} -- Directory where to store the xml.

        Returns:
            {str} -- Path to XML file.
        """

        # write file
        file_name = case_name + '.xml'
        file_path = os.path.join(working_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        flow1d_working_dir = os.path.join(working_dir, self.__1d_dir_name)
        f = open(file_path, 'w+')
        f.write(
            '<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n' +
            '<dimrConfig xmlns="http://schemas.deltares.nl/dimr"' +
            ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' +
            ' xsi:schemaLocation="http://schemas.deltares.nl/dimr' +
            ' http://content.oss.deltares.nl/schemas/dimr-1.2.xsd">\n' +
            '<documentation>\n' +
            '<fileVersion>1.2</fileVersion>\n' +
            '<createdBy>Deltares, Coupling Team</createdBy>\n' +
            '<creationDate>2019-03-28T08:04:13.0106499Z</creationDate>\n' +
            '</documentation>\n' +
            '<control>\n' +
            '<start name="rijn-flow-model" />\n' +
            '</control>\n' +
            '<component name="rijn-flow-model">\n' +
            '<library>cf_dll</library>\n' +
            '<workingDir>{}</workingDir>\n'.format(flow1d_working_dir) +
            '<inputFile>rijn-flow-model.md1d</inputFile>\n' +
            '</component>\n' +
            '</dimrConfig>\n')
        f.close()
        return file_path

    def __run_dimr_from_command(self, sobek_xml_location: str):
        """ Runs created xml with dimr script.

        Arguments:
            sobek_xml_location {str}
                -- Location of xml file that points to the working dir.
        """
        runner_dir = TestUtils.get_test_dir(self.__sobek_runner_dir_name)
        dimr_runner_path = os.path.join(runner_dir, self.__runner_script)
        if not os.path.exists(dimr_runner_path):
            raise IOError(
                'DIMR Runner not found at {}.'.format(dimr_runner_path))

        dimr_call = '{} {} -d 0 > out.txt 2>&1'.format(
                    dimr_runner_path, sobek_xml_location)
        try:
            os.system(dimr_call)
        except Exception as e_error:
            raise Exception(
                'Exception thrown while doing DIMR call.' +
                ' {}'.format(str(e_error)))

    def __get_2d_observations_file(self, fm_dir: str):
        """Finds the observation file for a 2d model

        Arguments:
            fm_dir {str} -- Directory for FM model.

        Raises:
            Exception: When path not found.

        Returns:
            {str} -- Path to observation file.
        """
        obs_2d_rel_path = 'resultaten\\FlowFM_his.nc'
        output_2d = os.path.join(fm_dir, obs_2d_rel_path)
        if not os.path.exists(output_2d):
            raise Exception(
                '2D output was not found at{}.'.format(output_2d))
        return output_2d

    def __get_1d_observations_file(self, sobek_dir: str):
        """Finds the observations.nc file generated from the sobek dimr run.

        Arguments:
            sobek_dir {str} -- Parent directory used for running dimr

        Returns:
            {str} -- File path where the observation file has been generated.
        """
        dflow1d_output = os.path.join(
            sobek_dir, self.__1d_dir_name + '\\output')
        if not os.path.exists(dflow1d_output):
            raise IOError(
                'Sobek output folder not created at' +
                ' {}.'.format(dflow1d_output))

        if not os.listdir(dflow1d_output):
            raise IOError(
                'No output generated at' +
                '{}.'.format(dflow1d_output))

        observations_file = os.path.join(dflow1d_output, 'observations.nc')
        if not os.path.exists(observations_file):
            raise IOError(
                'Observation file was not found at' +
                ' {}.'.format(observations_file))

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
        try:
            from tqdm import tqdm
        except:
            TestUtils.install_package('tqdm')
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
                # Avoid inserting points in file names.
                stat_fig_name = stat.replace('.', '_')
                fig_name = os.path.join(
                    fig_dir,
                    '{}_{}.png'.format(case_name, stat_fig_name))
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
        plt.close('all')
        return list_of_figures
