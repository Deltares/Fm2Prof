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
"""
Copyright (C) Stichting Deltares 2019. All rights reserved.

This file is part of the Fm2Prof.

The Fm2Prof is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.
"""

import os, sys, getopt, shutil
import configparser
from typing import Mapping, Sequence


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
               
        ini_file_params = {}
        
        try:
            ini_file_params = self.get_inifile_params(file_path)
        except Exception as e_info:
            raise Exception('It was not possible to extract ini parameters from the file {}. Exception thrown: {}'.format(file_path, str(e_info)))
        
        # Extract parameters
        self._output_dir = self._extract_output_dir(ini_file_params) # output directory path
        self._input_parameters = self._extract_input_parameters(ini_file_params) # dictionary which contains all input parameter values
        self._input_file_paths = self._extract_input_files(ini_file_params)

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
  