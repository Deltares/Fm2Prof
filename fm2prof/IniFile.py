"""
This module contains the IniFile class, which handles the main configuration file.
"""

# import from standard library
import os
import io
import json
import inspect
import configparser
from pydoc import locate
from logging import Logger
from typing import Mapping, AnyStr, Union, Dict, List, Type

# import from package
from fm2prof.common import FM2ProfBase

class InvalidConfigurationFileError(Exception):
    """ Raised when config file is not up to snot """
    pass


class ImportBoolType():
    """ Custom type to parse booleans """
    def __new__(cls, value):
        if value.lower().strip() == 'true':
            return True
        else:
            return False


class ImportListType():
    """ Custom type to parse list of comma separated ints """
    def __new__(cls, value):
        return list(map(int, value.strip('[]').split(',')))


class IniFile(FM2ProfBase):
    """
    This class provides all functionality to interact with the configuration file, e.g.:
        - reading
        - validating
        - retrieving parameters
        - retrieving files

    Parameters:
        file_path (str): path to filestring

    """
    __filePath = None
    __input_files_key = 'input'
    __input_parameters_key = 'parameters'
    __output_key = 'output'
    __output_directory_key = 'OutputDirectory'   
    __ini_keys = dict(
        map_file='2dmapoutput',
        css_file='crosssectionlocationfile',
        region_file='regionpolygonfile',
        section_file='sectionpolygonfile',
        export_mapfiles="exportmapfiles",
        css_selection="cssselection",
        classificationmethod="classificationmethod",
        sdfloodplainbase='sdfloodplainbase',
        sdstorage='sdstorage',
        transitionheight_sd='transitionheight_sd',
        number_of_css_points='number_of_css_points',
        minimum_width='minimum_width'
        )

    _output_dir = None
    _input_file_paths = None
    _input_parameters = None

    def __init__(self, file_path: str="", logger: Logger=None):
        """
        Initializes the object Ini File which contains the path locations of all
        parameters needed by the Fm2ProfRunner.

        The configuration file consists of three main sections: input, parameters and output.

            - **input** specifies paths to files. A check will be performed:
                - whether the file exists.
            - **parameters** specified parameters. A check will be performed:
                -  whether the parameters can be typeforced to the expected types
                -  whether the value conforms to allowed values
            - **output** specifies the output directory. A check will be performed:
                - whether the directory can be created

        Arguments:
            file_path {str} -- File path where the IniFile is located
        """
        super().__init__(logger=logger)
        self.__filePath = file_path
        self.set_logger_message(f'Received ini file: {self.__filePath} on {os.getcwd()}', 'debug')
        self._ini_template = self._get_template_ini()    # Template to fill defaults from
        self._configuration = self._get_template_ini()   # What will be used

        if isinstance(file_path, str):
            self.__filePath = file_path
            self.__fileDir = os.path.split(file_path)[0]
        else:
            # if no filepath, or filepath is StringIO object (for testing purposes)
            self.__filePath = None
            self.__fileDir = None
        if not(file_path is None or not file_path):
            self._read_inifile(file_path)

    @property
    def has_output_directory(self) -> bool:
        """
        Verifies if the output directory has been set and exists or not.
        Arguments:
            iniFile {IniFile} -- [description]
        Returns:
            True - the output_dir is set and exists.
            False - the output_dir is not set or does not exist.
        """
        if self.get_output_directory() is None:
            self.set_logger_message("No Output Set", 'warning')
            return False

        if not os.path.exists(self.get_output_directory()):
            try:
                os.makedirs(self.get_output_directory())
            except OSError:
                self.set_logger_message(
                    'The output directory {}, '.format(self.get_output_directory()) +
                    'could not be found neither created.', 'warning')
                return False

        return True

    def get_output_directory(self) -> AnyStr:
        """
        Use this method to return the output directory

        Returns:
            output directory (str)

        """

        return self._configuration['sections']['output'][self.__output_directory_key]['value']

    def get_ini_root(self, dirtype: str='relative') -> str:
        if dirtype=='relative':
            return self.__fileDir
        elif dirtype=='absolute':
            return os.path.join(os.getcwd(), self.__fileDir)

    def get_relative_path(self, path: str) -> str:
        return os.path.join(self.get_ini_root(), path)

    def get_input_file(self, key: str) -> AnyStr:
        """
        Use this method to retrieve the path to an input file

        Parameters:
            key (str): name of the input file
        """
        return self._get_from_configuration('input', key)

    def get_parameter(self, key: str) -> Union[str, bool, int, float]:
        """
        Use this method to return a parameter value
        """
        return self._get_from_configuration('parameters', key)
    
    def set_input_file(self, key: str, value=None) -> bool:
        """ Use this method to set a input files the configuration """
        self._set_config_value('input', key, value)
    
    def set_parameter(self, key, value) -> None:
        """ Use this method to set a key/value pair to the configuration """
        self._set_config_value('parameters', key, value)

    def _set_config_value(self, section, key, value) -> None:
        """ Use this method to set a input files the configuration """
        ckey = self._get_key_from_case_insensitive_input(section=section, key_in=key)
        if ckey:
            self._configuration['sections'][section][ckey]['value'] = value
            return True
        else:
            keys = self._configuration['sections'][section].keys()
            self.set_logger_message(f'Unknown {section}. Available keys: {list(keys)}')
            return False

    def set_output_directory(self, value: str) -> None:
        """ 
        Use this method to set the output directory 
        
        """
        case_name = self._get_valid_case_name(self.get_parameter('casename'), value)
        
        self._configuration['sections']['output'][self.__output_directory_key]['value'] = self._get_valid_output_dir(os.path.join(value, case_name))

    def print_configuration(self) -> str:
        """ Use this method to print a string of the configuration used """
        return self._print_configuration(self._configuration)

    def iter_parameters(self):
        """ Use this method to iterate through the names and values of all parameters """ 
        for parameter, content in self._configuration['sections'].get('parameters').items():
            yield parameter, content.get('type'), content.get('hint'), content.get('value')

    @staticmethod
    def _print_configuration(inputdict) -> str:
        f = io.StringIO()
        for sectionname, section in inputdict.get('sections').items():
            f.write(f'[{sectionname}]\n')
            for key, contents in section.items():
                f.write(f"{key:<30}= {str(contents.get('value')):<10}# {contents.get('hint')}\n")
            f.write('\n')
        return f.getvalue()

    def _get_key_from_case_insensitive_input(self, section, key_in):
        section=self._configuration['sections'][section]
        for key in section.keys():
            if key.lower() == key_in.lower():
                return key
        return ''

    def _get_from_configuration(self, section: str, key: str) -> Union[str, bool, int, float]:
        for parameter, content in self._configuration['sections'].get(section).items():
            if parameter.lower() == key.lower():
                return content.get('value')

    def _get_template_ini(self) -> Dict:
        # Open config file
        path, _ = os.path.split(inspect.getabsfile(IniFile))
        with open(os.path.join(path, 'configurationfile_template.json'), 'r') as f:
            default_ini = json.load(f)

        # parse all types
        return default_ini

    def _read_inifile(self, file_path: str):
        """
        Reads the inifile and extract all its parameters for later usage

        Parameters:
            file_path {str} -- File path where the IniFile is located
        """
        if file_path is None or not file_path:
            raise IOError('No ini file was specified and no data could be read.')
        try:
            if not os.path.exists(file_path):
                raise IOError('The given file path {} could not be found.'.format(file_path))
        except TypeError:
            if not isinstance(file_path, io.StringIO):
                raise IOError('Unknown file format entered')
        try:
            supplied_ini = self._get_inifile_params(file_path)
        except Exception as e_info:
            raise Exception('It was not possible to extract ini parameters from the file {}. Exception thrown: {}'.format(file_path, str(e_info)))

        # Compare supplied with default/expected inifile
        try:
            self._extract_input_files(supplied_ini)
        except Exception as e_info:
            self.set_logger_message('Unexpected error reading input files. Check config file', 'error')
        try:
            self._extract_parameters(supplied_ini)
        except Exception:
            self.set_logger_message('Unexpected error reading parameters. Check config file', 'error')
        try:
            self._extract_output_dir(supplied_ini)
        except Exception:
            self.set_logger_message('Unexpected error output parameters. Check config file', 'error')

    def _extract_parameters(self, supplied_ini: Mapping[str, list]):
        """ Extract InputParameters and convert values either integer or float from string

        Arguments:
            inifile_parameters {Mapping[str, list} -- Collection of parameters as read in the original file

        Returns:
            {Mapping[str, number]} -- Dictionary of mapped parameters to a either integer or float
        """
        try:
            inputsection = supplied_ini.get(self.__input_parameters_key)
        except KeyError:
            raise InvalidConfigurationFileError

        for key, value in inputsection.items():
            key_default, key_type = self._get_key_from_template(self.__input_parameters_key, key)
            try:
                parsed_value = key_type(value)
                self.set_parameter(key_default, parsed_value)
            except ValueError:
                self.set_logger_message(f'{key} could not be cast as {key_type}')
            except KeyError:
                pass

    def _extract_input_files(self, supplied_ini: Mapping[str, list]) -> None:
        """
        Extract and validates input files

        Parameters:
            inifile_parameters {Mapping[str, list]} -- Collection of parameters as read in the original file

        Returns:
            {Mapping[str, str]} -- new dict containing a normalized key (file name parameter) and its value.
        """
        try:
            inputsection = supplied_ini.get(self.__input_files_key)
        except KeyError:
            raise InvalidConfigurationFileError

        for key, value in inputsection.items():
            key_default, _ = self._get_key_from_template('input', key)

            if key_default is not None:
                if os.path.isfile(value):
                    self.set_input_file(key_default, value)
                elif os.path.isfile(os.path.join(self.__fileDir, value)):
                    self.set_input_file(key_default, os.path.join(self.__fileDir, value))
                else:
                    self.set_logger_message(f'Could not find input file for {key_default}, skipping', 'warning')
                    if key_default in ('2DMapOutput', 'CrossSectionLocationFile'):
                        self.set_logger_message(f'Could not find input file: {key_default}', 'error')
                        raise FileNotFoundError

    def _get_key_from_template(self, section, key) -> List[Union[str, Type]]:
        """ return list of lower case keys from default configuration files"""
        sectiondict = self._ini_template.get('sections').get(section)
        for entry in sectiondict:
            if key.lower() == entry.lower():
                return (entry, locate(sectiondict[entry].get('type')))
        # If not returned by now, key must be unknown
        self.set_logger_message(f"{key} is not a known key", 'warning')
        return [None, KeyError]

    def _extract_output_dir(self, supplied_ini: Mapping[str, list]):
        """
        Extract and validates output directory

        Parameters:
            supplied_ini {Mapping[str, list]} --  Collection of parameters as read in the original file

        Returns:
            {str} -- Normalized output dir path
        """
        try:
            outputsection = supplied_ini.get(self.__output_key)
        except KeyError:
            raise InvalidConfigurationFileError
        

        for key, value in outputsection.items():
            if key.lower() == self.__output_directory_key.lower():
                self.set_output_directory(value)
            else:
                self.set_logger_message(f'Unknown key {key} found in configuration file', 'warning')

    def _get_valid_output_dir(self, output_dir : str ):
        """
        Gets a normalized output directory path.

        Arguments:
            output_dir {str} -- Relative path to the configuration file.

        Returns:
            {str} -- Valid output directory path.
        """
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass
        return output_dir
        """
        if not output_dir:
            return os.getcwd()
        tmp_output_dir = output_dir.replace('/','\\')
        if '..' not in tmp_output_dir:
            return os.path.join(self.__fileDir, tmp_output_dir)
        return tmp_output_dir
        """

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
        output_dir = self._get_valid_output_dir(output_dir)
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
    def _output_dir(self):
        try:
            return self._configuration['sections']['output'][self.__output_directory_key]['value']
        except KeyError:
            return None
    
    @staticmethod
    def _get_inifile_params(file_path: str) -> Dict:
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
        if isinstance(file_path, io.StringIO):
            config.read_file(file_path)
        else:
            with open(file_path, 'r') as f:
                config.read_file(f)

        # Extract all sections and options
        for section in config.sections():
            ini_file_params[section.lower()] = {}
            for option in config.options(section):
                ini_file_params[section.lower()][option.lower()] = config.get(section, option).split(comment_delimter)[0].strip()

        return ini_file_params