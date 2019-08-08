"""
This module contains functions used for the emulation/reduction of 2D models to
1D models for Delft3D FM (D-Hydro).


Dependencies
------------------
Packages, between parenthesis known working version.

netCDF4 (1.2.1)
numpy (1.10.4)
pandas (0.17.1)
sklearn (0.15.2)
matplotlib (1.5.1)


Contact: K.D. Berends (koen.berends@deltares.nl, k.d.berends@utwente.nl)
"""

from fm2prof import Functions as FE
from fm2prof import Classes as CE
from fm2prof import sobek_export
from fm2prof import IniFile
from fm2prof.MaskOutputFile import MaskOutputFile

from typing import Mapping, Sequence
import datetime
import itertools
import os
import shutil
import logging
import time


class Fm2ProfRunner:
    __logger = None
    __iniFile = None
    __showFigures = False
    __saveFigures = False
    __version = None

    __map_file_key = 'fm_netcdfile'
    __css_file_key = 'crosssectionlocationfile'
    __gebiedsvakken_key = 'gebiedsvakken'
    __sectie_key = 'sectionfractionfile'

    def __init__(self, iniFilePath: str, version: float = None):
        """
        Initializes the private variables for the Fm2ProfRunner

        Arguments:
            iniFilePath {str}
                -- File path where the IniFile is located.
            version {float}
                -- Current version of the software, needs to be rethought.
        """
        self.__iniFile = IniFile.IniFile(iniFilePath)
        self.__version = version

        self.__create_logger()

    def run(self):
        """
        Runs the Fm2Prof functionality.
        """
        if self.__iniFile is None:
            self.__set_logger_message(
                'No ini file was specified and the run cannot go further.')
            return
        self.run_inifile(self.__iniFile)

    def run_inifile(self, iniFile: IniFile):
        """Runs the desired emulation from 2d to 1d given the mapfile
            and the cross section file.

        Arguments:
            iniFile {IniFile}
                -- Object containing all the information
                    needed to execute the program
        """
        if not self.__is_output_directory_set(iniFile):
            return

        # shorter local variables
        map_file = iniFile._input_file_paths.get(self.__map_file_key)
        css_file = iniFile._input_file_paths.get(self.__css_file_key)
        output_dir = iniFile._output_dir
        input_param_dict = iniFile._input_parameters

        # Add a log file
        self.__add_filehandler_to_logger(
            output_dir=output_dir,
            filename='fm2prof.log')
        self.__set_logger_message(
            'FM2PROF version {}'.format(self.__version))
        self.__set_logger_message('reading FM and cross-sectional data data')

        # Read FM model data
        fm2prof_fm_model_data = FE.read_fm2prof_input(map_file, css_file)
        fm_model_data = CE.FmModelData(fm2prof_fm_model_data)
        self.__set_logger_message(
            'finished reading FM and cross-sectional data data')

        # generate all cross-sections
        cross_sections = self._generate_cross_section_list(
            input_param_dict, fm_model_data)

        # The roughness tables in 1D model require
        # the same discharges on the rows.
        # This function interpolates to get the roughnesses
        # at the correct discharges.
        FE.interpolate_roughness(cross_sections)

        # Export cross sections
        self._export_cross_sections(cross_sections, output_dir)

        # Generate output geojson
        mask_points = [cs.get_mask_point() for cs in cross_sections]
        MaskOutputFile.write_mask_output_file(
            os.path.join(output_dir, 'mask_output.geojson'),
            mask_points)

    def _generate_cross_section_list(
            self,
            input_param_dict: Mapping[str, list],
            fm_model_data: CE.FmModelData):
        """ Generates cross sections based on the given fm_model_data

        Arguments:
            input_param_dict {Mapping[str, list]}
                -- Dictionary of parameters read from IniFile
            fm_model_data {CE.FmModelData}
                -- Class with all necessary data for generating Cross Sections

        Returns:
            {list} -- List of generated cross sections
        """
        cross_sections = list()
        if not fm_model_data or not input_param_dict:
            return cross_sections

        # Preprocess css from fm_model_data so it's easier to handle it.
        css_data_list = fm_model_data.css_data_list
        self.__logformatter.set_number_of_iterations(len(css_data_list))
        for css_data in css_data_list:
            generated_cross_section = self._generate_cross_section(
                css_data, input_param_dict, fm_model_data)
            if generated_cross_section is not None:
                cross_sections.append(generated_cross_section)
                self.__logformatter.start_new_iteration()

        return cross_sections

    def _generate_cross_section(
            self,
            css_data: dict,
            input_param_dict: Mapping[str, list],
            fm_model_data: CE.FmModelData):
        """Generates a cross section and configures its values based
        on the input parameter dictionary

        Arguments:
            css_data {dict}
                -- Dictionary of data for the current cross section
            input_param_dict {Mapping[str,list]}
                -- Dictionary with input parameters
            fm_model_data {CE.FmModelData}
                -- Data to assign to the new cross section

        Raises:
            Exception: If no css_data is given.
            Exception: If no input_param_dict is given.
            Exception: If no fm_model_data is given.

        Returns:
            {CE.CrossSection} -- New Cross Section
        """
        if css_data is None:
            raise Exception('No data was given to create a Cross Section')

        css_name = css_data.get('id')
        if not css_name:
            css_name = 'new_cross_section'

        if input_param_dict is None:
            raise Exception(
                'No input parameters (from ini file)' +
                ' given for new cross section {}'.format(css_name))

        if fm_model_data is None:
            raise Exception(
                'No FM data given for new cross section {}'.format(css_name))

        self.__logformatter.next_step()
        self.__set_logger_message(
            '{}'.format(css_name))

        # Create cross section
        created_css = self._get_new_cross_section(
            css_data=css_data,
            input_param_dict=input_param_dict)

        if created_css is None:
            raise Exception('No Cross-section could be generated')

        created_css.set_logger(self.__logger)
        self.__set_logger_message(
            'Initiated new cross-section')

        self._set_fm_data_to_cross_section(
            cross_section=created_css,
            input_param_dict=input_param_dict,
            fm_model_data=fm_model_data)

        self.__set_logger_message(
            'done')
        return created_css

    def _set_fm_data_to_cross_section(
            self,
            cross_section: CE.CrossSection,
            input_param_dict: Mapping[str, list],
            fm_model_data: CE.FmModelData,
            start_time=None):
        """Sets extra FM data to the given Cross Section

        Arguments:
            cross_section {CE.CrossSection}
                -- Given Cross Section.
            input_param_dict {Mapping[str, list]}
                -- Dictionary with input parameters.
            fm_model_data {CE.FmModelData}
                -- Data to assign to the new cross section.

        """

        if cross_section is None or fm_model_data is None:
            return

        # define fm_model_data variables
        time_independent_data = fm_model_data.time_independent_data
        edge_data = fm_model_data.edge_data
        time_dependent_data = fm_model_data.time_dependent_data
        css_name = cross_section.name
        try:
            # Retrieve FM data for cross-section
            fm_data = FE.get_fm2d_data_for_css(
                cross_section.name,
                time_independent_data,
                edge_data,
                time_dependent_data)

            self.__set_logger_message(
                'Retrieved data for cross-section')

            # Build cross-section
            self.__set_logger_message('Start building geometry')
            cross_section.build_from_fm(fm_data=fm_data)

            self.__set_logger_message(
                'Cross-section derived, starting correction.....')

            # Delta-h correction
            self._calculate_css_correction(
                input_param_dict, cross_section)

            # Reduce number of points in cross-section
            self._reduce_css_points(
                input_param_dict, cross_section)

            # assign roughness
            self.__set_logger_message('Starting computing roughness tables')
            cross_section.assign_roughness(fm_data)

            self.__set_logger_message(
                'computed roughness')

        except Exception as e_info:
            self.__set_logger_message(
                'Exception while setting cross-section' +
                ' {} details, {}'.format(css_name, str(e_info)))

    def _get_new_cross_section(
            self,
            css_data: Mapping[str, str],
            input_param_dict: Mapping[str, str]):
        """Creates a cross section with the given input param dictionary.

        Arguments:
            css_data {Mapping[str, str]}
                -- FM Model data for cross section.
            input_param_dict {Mapping[str, str]}
                -- Dictionary with parameters for Cross Section.

        Returns:
            {CE.CrossSection} -- New cross section object.
        """
        # Get id data and id index
        if not css_data:
            return None

        css_data_id = css_data.get('id')
        if not css_data_id:
            return None

        # Get remainig data
        css_data_length = css_data.get('length')
        css_data_location = css_data.get('xy')
        css_data_branch_id = css_data.get('branchid')
        css_data_chainage = css_data.get('chainage')

        if (css_data_length is None or
                css_data_location is None or
                css_data_branch_id is None or
                css_data_chainage is None):
            return None

        try:
            css = CE.CrossSection(
                input_param_dict,
                name=css_data_id,
                length=css_data_length,
                location=css_data_location,
                branchid=css_data_branch_id,
                chainage=css_data_chainage)
        except Exception as e_info:
            self.__set_logger_message(
                'Exception thrown while creating cross-section ' +
                '{}, message: {}'.format(css_data_id, str(e_info)))
            return None

        return css

    def _export_cross_sections(self, cross_sections: list, output_dir: str):
        """Exports all cross sections to the necessary file formats

        Arguments:
            cross_sections {list}
                -- List of created cross sections
            output_dir {str}
                -- target directory where to export all the cross sections
        """
        if not cross_sections:
            return

        if not output_dir or not os.path.exists(output_dir):
            return

        # File paths
        css_location_ini_file = os.path.join(
            output_dir, 'CrossSectionLocations.ini')
        css_definitions_ini_file = os.path.join(
            output_dir, 'CrossSectionDefinitions.ini')

        csv_geometry_file = output_dir + '\\geometry.csv'
        csv_roughness_file = output_dir + '\\roughness.csv'

        csv_geometry_test_file = output_dir + '\\geometry_test.csv'
        csv_roughness_test_file = output_dir + '\\roughness_test.csv'

        csv_volumes_file = output_dir + '\\volumes.csv'

        # export all cross-sections
        try:
            sobek_export.export_crossSectionLocations(
                cross_sections, file_path=css_location_ini_file)

            sobek_export.export_geometry(
                cross_sections,
                file_path=css_definitions_ini_file,
                fmt='dflow1d')
            sobek_export.export_geometry(
                cross_sections, file_path=csv_geometry_file, fmt='sobek3')
            sobek_export.export_geometry(
                cross_sections,
                file_path=csv_geometry_test_file,
                fmt='testformat')

            sobek_export.export_roughness(
                cross_sections,
                file_path=csv_roughness_file,
                fmt='sobek3')
            sobek_export.export_roughness(
                cross_sections,
                file_path=csv_roughness_test_file,
                fmt='testformat')

            sobek_export.export_volumes(
                cross_sections,
                file_path=csv_volumes_file)
        except Exception as e_info:
            self.__set_logger_message(
                'An error was produced,' +
                ' not all output files might be exported. ' +
                '{}'.format(str(e_info)))

        self.__set_logger_message('Exported output files, FM2PROF finished')

    def _calculate_css_correction(
            self,
            input_param_dict: Mapping[str, float],
            css: CE.CrossSection):
        """Calculates the Cross Section correction if needed.

        Arguments:
            input_param_dict {Mapping[str,float]} -- [description]
            css {CE.CrossSection} -- [description]

        """
        if not css:
            self.__set_logger_message('No Cross Section was provided.')
            return

        # Get value, it should already come as an integer.
        sd_storage_key = 'sdstorage'
        sd_storage_value = input_param_dict.get(sd_storage_key)

        # Get value, it should already come as a float.
        sd_transition_key = 'transitionheight_sd'
        sd_transition_value = input_param_dict.get(sd_transition_key)

        # Check if it should be corrected.
        if sd_storage_value == 1:
            try:
                css.calculate_correction(sd_transition_value)
                self.__set_logger_message(
                    'correction finished')
            except Exception as e_error:
                e_message = str(e_error)
                self.__set_logger_message(
                    'Exception thrown ' +
                    'while trying to calculate the correction. ' +
                    '{}'.format(e_message))

    def _reduce_css_points(
            self,
            input_param_dict: Mapping[str, str],
            css: CE.CrossSection):
        """Returns a valid value for the number of css points read from ini file.

        Arguments:
            input_param_dict {Mapping[str, str]}
                -- Dictionary of elements read in the Ini File

        Returns:
            {Integer} -- Valid number of css points (default: 20)
        """
        num_css_pt_key = 'number_of_css_points'
        # Will return None if not found / defined.
        num_css_pt_value = input_param_dict.get(num_css_pt_key)
        # 20 is our default value for css_pt_value
        css_pt_value = 20
        try:
            # try to reduce points.
            css_pt_value = int(num_css_pt_value)
        except:
            self.__set_logger_message(
                'number_of_css_points given is not an int;' +
                ' Will use default (20) instead')

        try:
            css.reduce_points(n=css_pt_value, verbose=False)

        except Exception as e_error:
            e_message = str(e_error)
            self.__set_logger_message(
                'Exception thrown while trying to reduce the css points. ' +
                '{}'.format(e_message))

    def __set_logger_message(self, err_mssg: str, level='info'):
        """Sets message to logger if this is set.

        Arguments:
            err_mssg {str} -- Error message to send to logger.
        """
        if not self.__logger:
            return

        if level.lower() not in [
                'info', 'debug', 'warning', 'error', 'critical']:
            self.__logger.error(
                "{} is not valid logging level.".format(level.lower()))

        if level.lower() == 'info':
            self.__logger.info(err_mssg)
        elif level.lower() == 'debug':
            self.__logger.debug(err_mssg)
        elif level.lower() == 'warning':
            self.__logger.warning(err_mssg)
        elif level.lower() == 'error':
            self.__logger.error(err_mssg)
        elif level.lower() == 'critical':
            self.__logger.critical(err_mssg)

    def __get_time_stamp_seconds(self, start_time: datetime):
        """Returns a time stamp with the time difference

        Arguments:
            start_time {datetime} -- Initial date time

        Returns:
            {float} -- difference of time between start and now in seconds
        """
        time_now = datetime.datetime.now()
        time_difference = time_now - start_time
        return time_difference.total_seconds()

    def __is_output_directory_set(self, iniFile: IniFile):
        """
        Verifies if the output directory has been set and exists or not.
        Arguments:
            iniFile {IniFile} -- [description]
        Returns:
            True - the output_dir is set and exists.
            False - the output_dir is not set or does not exist.
        """
        if iniFile is None or iniFile._output_dir is None:
            print("The output directory must be set before running.")
            return False

        if not os.path.exists(iniFile._output_dir):
            try:
                os.makedirs(iniFile._output_dir)
            except:

                print(
                    'The output directory {}, '.format(iniFile._output_dir) +
                    'could not be found neither created.')
                return False

        return True

    def __generate_output(
            self, output_directory: str, fig, figType: str, name: str):
        if not self.__saveFigures:
            return

        plotLocation = output_directory + '\\{0}_{1}.png'.format(name, figType)
        fig.savefig(plotLocation)
        self.__set_logger_message(
            'Saved {} for {} plot in {}.'.format(name, figType, plotLocation))

        return

    def __add_filehandler_to_logger(self, output_dir, filename='fm2prof.log'):
        # create file handler
        fh = logging.FileHandler(os.path.join(output_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.__logformatter)
        self.__logger.addHandler(fh)

    def __create_logger(self):
        # Create logger
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)

        # create formatter
        self.__logformatter = CE.ElapsedFormatter()

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.__logformatter)
        self.__logger.addHandler(ch)
