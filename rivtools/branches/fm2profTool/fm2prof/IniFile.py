"""
inifile
"""

import os, sys, getopt, shutil
import configparser
from typing import Mapping, Sequence, AnyStr, Union, Dict

class IniFile:
    __logger = None
    __filePath = None

    # region Private parameter
    __input_files_key = 'InputFiles'
    __input_parameters_key = 'InputParameters'
    __output_directory_key = 'OutputDirectory'

    __ini_keys = dict(
        map_file='fm_netcdfile',
        css_file='crosssectionlocationfile',
        region_file='regionpolygonfile',
        section_file='sectionpolygonfile',
        export_mapfiles="exportmapfiles",
        css_selection="cssselection",
        classificationmethod="classificationmethod")
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
               
        ini_file_params = {}
        
        try:
            ini_file_params = self.get_inifile_params(file_path)
        except Exception as e_info:
            raise Exception('It was not possible to extract ini parameters from the file {}. Exception thrown: {}'.format(file_path, str(e_info)))
        
        # Extract parameters
        self._output_dir = self._extract_output_dir(ini_file_params) # output directory path
        self._input_parameters = self._extract_input_parameters(ini_file_params) # dictionary which contains all input parameter values
        self._input_file_paths = self._extract_input_files(ini_file_params)

    def get_output_directory(self) -> AnyStr:
        """ 
        Use this method to return the output directory 
        
        Returns:
            output directory (str)
        
        """

        return self._output_dir
    
    @staticmethod
    def get_inifile_params(file_path : str):
        """Extracts the parameters from an ini file.
        
        Arguments:
            file_path {str} -- Ini file location
        
        Returns:
            {array} -- List of sections containing list of options
        """
        ini_file_params = {}
        comment_delimter = '#'
        
        # Create config parser (external class)
        config = configparser.ConfigParser()
        config.read(file_path)
        
        # Extract all sections and options
        for section in config.sections():
            ini_file_params[section] = {}
            for option in config.options(section):
                ini_file_params[section][option] = config.get(section, option).split(comment_delimter)[0].strip()
        
        return ini_file_params

    def get_input_file(self, key: str) -> AnyStr:
        return self._input_file_paths.get(self.__ini_keys[key])

    def get_parameter(self, key: str) -> Union[str, bool, int, float]:
        """ 
        Use this method to return a parameter value
        """
        return self._input_parameters.get(self.__ini_keys[key])

    def get_parameters(self) -> Dict:
        return self._input_parameters


    def _extract_input_parameters(self, inifile_parameters : Mapping[str, list]):
        """ Extract InputParameters and convert values either integer or float from string
       
        Arguments:
            inifile_parameters {Mapping[str, list} -- Collection of parameters as read in the original file
        
        Returns:
            {Mapping[str, number]} -- Dictionary of mapped parameters to a either integer or float
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
                    input_parameters[sub] = int(float_value)
                else: # if float
                    input_parameters[sub] = float_value
            except ValueError:
                try:
                     input_parameters[sub] = list(map(int, parameter_value.split(',')))
                except ValueError:
                    input_parameters[sub] = None
        return input_parameters
        
    def _extract_input_files( self, inifile_parameters : Mapping[str, list]):
        """ Extract input file information from the dictionary

        Arguments:
            inifile_parameters {Mapping[str, list]} -- Collection of parameters as read in the original file
        
        Returns:
            {Mapping[str, str]} -- new dict containing a normalized key (file name parameter) and its value.
        """
        file_parameters = {}
        input_files_parameters = inifile_parameters.get(self.__input_files_key)
        if input_files_parameters is None:
            return file_parameters

        for file_parameter in input_files_parameters:
            new_key = file_parameter.lower()
            file_parameters[new_key] = input_files_parameters.get(file_parameter)
        
        return file_parameters
    
    def _extract_output_dir(self, inifile_parameters: Mapping[str, list]):
        """Extract output directory infomation from the dictionary
        
        Arguments:
            inifile_parameters {Mapping[str, list]} --  Collection of parameters as read in the original file
        
        Returns:
            {str} -- Normalized output dir path
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
        """Gets a normalized output directory path.
        
        Arguments:
            output_dir {str} -- Relative path to the output directory.
        
        Returns:
            {str} -- Valid output directory path.
        """
        if not output_dir:
            return os.getcwd()
        tmp_output_dir = output_dir.replace('/','\\')
        if '..' not in tmp_output_dir:
            return os.path.join(os.getcwd(), tmp_output_dir)
        return tmp_output_dir

    def _get_valid_case_name(self, case_name : str, output_dir : str):
        """Gets a valid case name to avoid duplication of directories
        
        Arguments:
            case_name {str} -- Given current case name
            output_dir {str} -- Target output directory path
        
        Returns:
            {str} -- Unique case name
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
    
    @property
    def has_output_directory(self):
        """
        Verifies if the output directory has been set and exists or not.
        Arguments:
            iniFile {IniFile} -- [description]
        Returns:
            True - the output_dir is set and exists.
            False - the output_dir is not set or does not exist.
        """
        if self._output_dir is None:
            print("The output directory must be set before running.")
            return False

        if not os.path.exists(self._output_dir):
            try:
                os.makedirs(self._output_dir)
            except:

                print(
                    'The output directory {}, '.format(self._output_dir) +
                    'could not be found neither created.')
                return False

        return True