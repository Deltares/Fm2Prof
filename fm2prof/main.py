#! /usr/bin/env python
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

__version__ = 1.1
__revision__ = 2

# region // imports
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import datetime
import seaborn as sns
import itertools
import configparser

from fm2prof import Functions as FE
from fm2prof import Classes as CE
from fm2prof import sobek_export

from typing import Mapping, Sequence

import os, sys, getopt
# endregion

class IniFile:
    __logger = None
    __filePath = None

    # region Private parameter
    __input_files_key = 'InputFiles'
    __input_parameters_key = 'InputParameters'
    __output_directory_key = 'OutputDirectory'
    # endregion

    # region Public parameters
    _output_dir = None
    _input_file_paths = None
    _input_parameters = None
    # endregion

    def __init__(self, file_path: str):
        """
        Initializes the object Ini File which contains the path locations of all
        parameters needed by the Fm2ProfRunner
        
        Arguments:
            file_path {str} -- File path where the IniFile is located
        """
        self.__filePath = file_path
        if not(file_path is None or not file_path):
            self._read_inifile(file_path)
    
    def _read_inifile(self, file_path : str):
        """
        Reads the inifile and extract all its parameters for later usage by the 
        Fm2ProfRunner
        
        Arguments:
            file_path {str} -- File path where the IniFile is located
        """
        if file_path is None or not file_path:
            raise IOError('No ini file was specified and no data could be read.')
        if not os.path.exists(file_path):
            raise IOError('The given file path {} could not be found.'.format(file_path))
        
        config = configparser.ConfigParser()
        config.read(file_path)
        
        ini_file_params = {}
        for section in config.sections():
            ini_file_params[section] = {}
            for option in config.options(section):
                ini_file_params[section][option] = config.get(section, option).split('#')[0].strip()
        
        # Extract parameters
        self._output_dir = self._extract_output_dir(ini_file_params) # output directory path
        self._input_parameters = self._extract_input_parameters(ini_file_params) # dictionary which contains all input parameter values
        self._input_file_paths = self._extract_input_files(ini_file_params)

    def _extract_input_parameters(self, inifile_parameters : Mapping[str, str]):
        """
        Extract InputParameters and convert values either integer or float from string
        Returns:
            Dictionary of mapped parameters to a either integer or float
        Arguments:
            inifile_parameters {Mapping[str, str]} -- Collection of parameters as read in the original file
        """
        if inifile_parameters is None:
            return None

        input_parameters = inifile_parameters.get(self.__input_parameters_key)
        if input_parameters is None:
            return None

        for sub in input_parameters:
            parameter_value = input_parameters.get(sub)
            try:
                float_value = float(parameter_value)
                if float_value.is_integer(): # if integer
                    input_parameters[sub] = int(parameter_value)
                else: # if float
                    input_parameters[sub] = float_value
            except ValueError:
                input_parameters[sub] = None
        return input_parameters
        
    def _extract_input_files( self, inifile_parameters : Mapping[str, str]):
        """
        Extract input file information from the dictionary
        Returns:
            new Mapping[str,str] containing a normalized key (file name parameter) and its value.
        Arguments:
            inifile_parameters {Mapping[str, str]} -- Collection of parameters as read in the original file
        """
        file_parameters = {}
        input_files_parameters = inifile_parameters.get(self.__input_files_key)
        if input_files_parameters is None:
            return file_parameters

        for file_parameter in input_files_parameters:
            new_key = file_parameter.lower()
            file_parameters[new_key] = input_files_parameters.get(file_parameter)
        
        return file_parameters
    
    def _extract_output_dir(self, inifile_parameters : Mapping[str, str]):
        """
        Extract output directory infomation from the dictionary
        
        Arguments:
            inifile_parameters {Mapping[str, str]} -- Collection of parameters as read in the original file
        """
        output_parameters = inifile_parameters.get(self.__output_directory_key)
        if output_parameters is None:
            return None
        
        # Get outputdir parameter
        output_dir_key = 'outputdir'
        output_dir = output_parameters.get(output_dir_key, '')        
        output_dir = self._get_valid_output_dir(output_dir)        

        # Get casename parameter
        case_name_key = 'casename'
        case_name = output_parameters.get(case_name_key, '')
        case_name = self._get_valid_case_name(case_name, output_dir)

        output_dir =  os.path.join(output_dir, case_name)
        
        return output_dir.replace("\\","/")

    def _get_valid_output_dir(self, output_dir : str ):
        """Returns a valid output directory path
        Reteurns :
            valid directory path.
        Arguments:
            output_dir {str} -- Relative path to the output directory.
        """
        if not output_dir:
            return os.getcwd()
        tmp_output_dir = output_dir.replace('/','\\')
        if '..' not in tmp_output_dir:
            return os.path.join(os.getcwd(), tmp_output_dir)
        return tmp_output_dir

    def _get_valid_case_name(self, case_name : str, output_dir : str):
        """
        Returns a valid case name to avoid duplication of directories
        
        Arguments:
            case_name {str} -- Given case name.
            output_dir {str} -- Target parent directory
        """
        case_num = 1
        default_name = 'CaseName'
        # If no case name was given, assign default.
        if not case_name:
            case_name = default_name

        # Set an index to the case
        case_name_tmp = case_name + '{:02d}'.format(case_num)
        relative_path = case_name_tmp # by default use the current directory
        if output_dir:
            relative_path = os.path.join(output_dir, case_name_tmp)
        
        # Ensure the case name is unique.
        while os.path.isdir(relative_path):
            case_num += 1
            case_name_tmp = case_name + '{:02d}'.format(case_num)
            # update relative_path and check is not present
            relative_path = os.path.join(output_dir, case_name_tmp)
            
        return case_name_tmp
  

class Fm2ProfRunner :  
    __logger = None    
    __iniFile = None
    __showFigures = False
    __saveFigures = False

    __map_file_key = 'fm_netcdfile'
    __css_file_key = 'crosssectionlocationfile'    
    __gebiedsvakken_key = 'gebiedsvakken'
    __sectie_key = 'sectionfractionfile'

    def __init__(self, iniFilePath : str):
        """
        Initializes the private variables for the Fm2ProfRunner        

        Arguments:
            iniFilePath {str} -- File path where the IniFile is located
        """
        self.__iniFile = IniFile(iniFilePath)

    def run(self):
        """
        Runs the Fm2Prof functionality.
        """
        if self.__iniFile is None:
            self.__set_logger_message('No ini file was specified and the run cannot go further.')
            return
        self.run_inifile(self.__iniFile)
      
    def run_inifile(self, iniFile : IniFile):
        """Runs the desired emulation from 2d to 1d given the mapfile and the cross section file.
        
        Arguments:
            iniFile {IniFile} -- Object containing all the information needed to execute the program
        """
        if not self.__is_output_directory_set(iniFile):
            return

        # shorter local variables
        map_file = iniFile._input_file_paths.get(self.__map_file_key)
        css_file = iniFile._input_file_paths.get(self.__css_file_key)
        output_dir = iniFile._output_dir
        input_param_dict = iniFile._input_parameters

        # Add a log file
        self.__logger = CE.Logger(output_dir)
        self.__set_logger_message('FM2PROF version {}\n=============================='.format(__version__))
        self.__set_logger_message('reading FM and cross-sectional data data')

        # Create an empty list. New cross-sections will be appended to this list. 
        cross_sections = list()

        # Read FM model data
        fm2prof_fm_model_data = FE.read_fm2prof_input(map_file, css_file)
        fm_model_data = CE.FmModelData(fm2prof_fm_model_data)
        self.__set_logger_message('finished reading FM and cross-sectional data data')

        # generate all cross-sections
        self._generate_cross_section_list(input_param_dict, fm_model_data)

        # The roughness tables in 1D model require the same discharges on the rows. 
        # This function interpolates to get the roughnesses at the correct discharges
        FE.interpolate_roughness(cross_sections)
        self._export_cross_sections(cross_sections, output_dir)

    def _generate_cross_section_list(self, input_param_dict : Mapping[str,str], fm_model_data : CE.FmModelData):
        """Generates cross sections based on the given fm_model_data
        
        Arguments:
            fm_model_data {CE.FmModelData} -- Class with all necessary data for generating Cross Sections
        """
        cross_sections = list()
        if not fm_model_data or not input_param_dict:
            return cross_sections

        css_data = fm_model_data.css_data
        css_data_name_list = css_data.get('id')
      
        for name in css_data_name_list:
            generated_cross_section = self._generate_cross_section(name, input_param_dict, fm_model_data)
            if generated_cross_section is not None:
                cross_sections.append(generated_cross_section)

        return cross_sections
    
    def _generate_cross_section(self, css_name : str, input_param_dict : Mapping[str,str], fm_model_data : CE.FmModelData):
        """Generates a cross section and configures its values based on the input parameter dictionary
        
        Arguments:
            css_name {str} -- Name for the new Cross Section
            input_param_dict {Mapping[str,str]} -- Dictionary with input parameters
            fm_model_data {CE.FmModelData} -- Data to assign to t he new cross section
        
        Raises:
            Exception: If no input_param_dict is given.
            Exception: If no fm_model_data is given
        
        Returns:
            {CE.CrossSection} -- New Cross Section
        """
        if not css_name:
            css_name = 'new_cross_section'

        if input_param_dict is None:
            raise Exception('No input parameters (from ini file) given for new cross section {}'.format(css_name))

        if fm_model_data is None:
            raise Exception('No FM data given for new cross section {}'.format(css_name))

        # define fm_model_data variables
        css_data = fm_model_data.css_data
        time_independent_data = fm_model_data.time_independent_data
        edge_data = fm_model_data.edge_data
        time_dependent_data = fm_model_data.time_dependent_data
        
        # time stamp start
        start_time = datetime.datetime.now()
        time_stamp = datetime.datetime.strftime(start_time, '%I:%M%p')        
        self.__set_logger_message('{} :: cross-section {}'.format(time_stamp, css_name))

        # Create cross section
        created_css = self._get_new_cross_section(css_name, input_param_dict = input_param_dict, css_data = css_data) 
        self.__set_logger_message('T+ %.2f :: initiated new cross-section %s' % (self.__get_time_stamp_seconds(start_time), css_name))

        try:            
            # Retrieve FM data for cross-section
            fm_data = FE.get_fm2d_data_for_css(created_css.name,
                                                time_independent_data,
                                                edge_data,
                                                time_dependent_data)
            
            self.__set_logger_message('T+ %.2f :: retrieved data for css %s' % (self.__get_time_stamp_seconds(start_time), css_name))

            # Build cross-section
            created_css.build_from_fm(fm_data=fm_data)
            self.__set_logger_message('T+ %.2f :: cross-section derived, starting correction.....' % (self.__get_time_stamp_seconds(start_time)))

            # Delta-h correction
            self._calculate_css_correction(input_param_dict, created_css, start_time)

            # Reduce number of points in cross-section
            self._reduce_css_points(input_param_dict, created_css, start_time)

            # assign roughness
            created_css.assign_roughness(fm_data)
            self.__set_logger_message('T+ %.2f :: computed roughness' % (self.__get_time_stamp_seconds(start_time)))
        
        except Exception as e_info:
            self.__set_logger_message('Exception while setting cross-section {} details, {}'.format(css_name, str(e_info)))

        self.__set_logger_message('cross-section {} generated in {:.2f} seconds'.format(css_name, self.__get_time_stamp_seconds(start_time)))
        return created_css

    def _get_new_cross_section(self, name:str, input_param_dict : Mapping[str, str], css_data : Mapping[str, str]):
        """Creates a cross section with the given input param dictionary.
        
        Arguments:
            name {str} -- Name for the new cross section.
            input_param_dict {Mapping[str, str]} -- Dictionary with parameters for Cross Section.
            css_data {Mapping[str, str]} -- FM Model data for cross section.
        
        Returns:
            {CE.CrossSection} -- New cross section object.
        """ 
        if not name:
            name = 'new_cross_section'        

        # Get id data and id index
        if not css_data:
            return None
        
        css_data_id = css_data.get('id')
        if not css_data_id:
            return None

        css_index = -1
        try:
            css_index = css_data_id.index(name)
        except ValueError as ve:
            self.__set_logger_message('Exception thrown while creating cross-section {}, message: {}'.format(name, str(ve)))
            return None
        
        # Get remainig data
        css_data_length = css_data.get('length')
        css_data_location = css_data.get('xy')
        css_data_branch_id = css_data.get('branchid')    
        css_data_chainage = css_data.get('chainage')

        if (css_data_length is None 
            or css_data_location is None 
            or css_data_branch_id is None 
            or css_data_chainage is None):
            return None

        try:
            css = CE.CrossSection(input_param_dict, 
                name=name, 
                length=css_data_length[css_index], 
                location=css_data_location[css_index],
                branchid=css_data_branch_id[css_index],
                chainage=css_data_chainage[css_index])
        except Exception as e_info:
            self.__set_logger_message('Exception thrown while creating cross-section {}, message: {}'.format(name, str(e_info)))
            return None

        return css

    def _export_cross_sections(self, cross_sections : list, output_dir: str):
        """Exports all cross sections to the necessary file formats
        
        Arguments:
            cross_sections {list} -- List of created cross sections
            output_dir {str} -- target directory where to export all the cross sections
        """
        if not cross_sections:
            return
        
        if not output_dir:
            output_dir = 'exported_cross_sections'
        # File paths
        css_location_ini_file = os.path.join(output_dir, 'CrossSectionLocations.ini')
        css_definitions_ini_file = os.path.join(output_dir, 'CrossSectionDefinitions.ini')
        
        csv_geometry_file = output_dir + '\\geometry.csv'
        csv_roughness_file = output_dir + '\\roughness.csv'
        
        csv_geometry_test_file = output_dir + '\\geometry_test.csv'
        csv_roughness_test_file = output_dir + '\\roughness_test.csv'

        csv_volumes_file = output_dir + '\\volumes.csv'

        # export all cross-sections         
        sobek_export.export_crossSectionLocations(cross_sections, file_path= css_location_ini_file )
        
        sobek_export.export_geometry(cross_sections, file_path = css_definitions_ini_file, fmt='dflow1d')
        sobek_export.export_geometry(cross_sections, file_path = csv_geometry_file, fmt = 'sobek3')
        sobek_export.export_geometry(cross_sections, file_path = csv_geometry_test_file, fmt='testformat')
        
        sobek_export.export_roughness(cross_sections, file_path = csv_roughness_file , fmt='sobek3')
        sobek_export.export_roughness(cross_sections, file_path = csv_roughness_test_file, fmt='testformat')
        
        sobek_export.export_volumes(cross_sections, file_path = csv_volumes_file)
        
        self.__set_logger_message('Exported output files, FM2PROF finished')

    def _calculate_css_correction(self, input_param_dict : Mapping[str,str], css : CE.CrossSection, start_time : float):
        """Calculates the Cross Section correction if needed.
        
        Arguments:
            input_param_dict {Mapping[str,str]} -- [description]
            css {CE.CrossSection} -- [description]
            start_time {float} -- [description]
        """
        sd_storage_key = 'sdstorage'
        sd_storage_value = input_param_dict.get(sd_storage_key, 0)
        # Verify the obtained value is an integer.
        if not sd_storage_value.isdigit():
            self.__set_logger_message('SDStorage value given is not an integer.')
            return
        # Check if it should be corrected.
        sd_storage_value = int(sd_storage_value)
        if sd_storage_value == 1:
            try:
                css.calculate_correction()
                self.__set_logger_message('T+ %.2f :: correction finished' % (datetime.datetime.now()-start_time).total_seconds())
            except Exception as e_error:
                e_message = str(e_error)
                self.__set_logger_message('Exception thrown while trying to calculate the correction. {}'.format(e_message))

    def _reduce_css_points(self, input_param_dict : Mapping[str, str], css : CE.CrossSection, start_time : float ):
        """Returns a valid value for the number of css points read from ini file.
        
        Arguments:
            input_param_dict {Mapping[str, str]} -- Dictionary of elements read in the Ini File
        
        Returns:
            {Integer} -- Valid number of css points (default: 20)
        """
        num_css_pt_key = 'number_of_css_points'
        num_css_pt_value = input_param_dict.get(num_css_pt_key, '') # Will return None if not found / defined.
        css_pt_value = 20 # 20 is our default value for css_pt_value
        if num_css_pt_value.isdigit():
            css_pt_value = int(num_css_pt_value)
        try:
            css.reduce_points( n = css_pt_value, verbose = False)
            self.__set_logger_message('T+ {:.2f} :: simplified cross-section to {:d} points'.format((datetime.datetime.now()-start_time).total_seconds(), css_pt_value))
        except Exception as e_error:
            e_message = str(e_error)
            self.__set_logger_message('Exception thrown while trying to reduce the css points. {}'.format(e_message))

    def __set_logger_message(self, err_mssg : str):
        """Sets message to logger if this is set.
        
        Arguments:
            err_mssg {str} -- Error message to send to logger.
        """
        if not self.__logger:
            return
        self.__logger.write(err_mssg)
    
    def __get_time_stamp_seconds(self, start_time : datetime):
        """Returns a time stamp with the time difference
        
        Arguments:
            start_time {datetime} -- Initial date time
        
        Returns:
            {float} -- difference of time between start and now in seconds
        """
        time_now = datetime.datetime.now()
        time_difference = time_now - start_time
        return time_difference.total_seconds()
    
    def __is_output_directory_set(self, iniFile : IniFile):
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
                
                print("The output directory {0}, could not be found neither created.".format(iniFile._output_dir))
                return False
        
        return True

    def __generate_output(self, output_directory : str, fig, figType : str, name : str):
        if not self.__saveFigures:
            return
        
        plotLocation = output_directory + '\\{0}_{1}.png'.format(name, figType)
        fig.savefig(plotLocation)
        self.__set_logger_message('Saved {0} for {1} plot in {2}.'.format(name, figType, plotLocation))
        
        return
    

# region // Main helpers

def __report_expected_arguments(reason):
    print('main.py -i <ini_file>')
    sys.exit("Error: {0}".format(reason))

def __is_input(argument):
    # Argument array has two elements
    # argument[0] = type
    # argument[1] = value
    argType = argument[0]
    return argType in ("-i", "--ifile")

def __is_output(argument):
    # Argument array has two elements
    # argument[0] = type
    # argument[1] = value
    argType = argument[0]
    return argType in ("-o", "--ofile")

# endregion

def main(argv):    
    """ Main class, should contain three input arguments and one output.
    Otherwise the execution will end with an error.
    
    Arguments:
        argv {[str]} -- default input from command line
    """
    # First try to pars the arguments
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        __report_expected_arguments("Arguments could not be retrieved.")
   
    # Check if number of arguments match the expectation.
    if len(opts) != 1:
        __report_expected_arguments("Not all arguments were given.")
    
    # Check if input parameters are in expected order
    if not __is_input(opts[0]):
        __report_expected_arguments("The first argument should be an input file.\n Given: {0}\n".format(opts[0]))
    
    # Run Fm2Prof with given arguments
    ini_file_path = opts[0][1]
    runner = Fm2ProfRunner(ini_file_path)
    runner.run()


if __name__ == '__main__':
    main(sys.argv[1:])