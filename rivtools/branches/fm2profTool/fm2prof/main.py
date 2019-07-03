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
import numbers

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

    def __init__(self, filePath: str):
        """
        Initializes the object Ini File which contains the path locations of all
        parameters needed by the Fm2ProfRunner
        
        Arguments:
            filePath {str} -- File path where the IniFile is located
        """
        self.__filePath = filePath
        if not(filePath is None or not filePath):
            self._read_inifile(filePath)
    
    def _read_inifile(self, filePath : str):
        """
        Reads the inifile and extract all its parameters for later usage by the 
        Fm2ProfRunner
        
        Arguments:
            filePath {str} -- File path where the IniFile is located
        """
        if filePath is None or not filePath:
            raise Exception('No ini file was specified and no data could be read.')
        
        config = configparser.ConfigParser()
        config.read(filePath)
        
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
                if isinstance(parameter_value, numbers.Integral): # if integer
                    input_parameters[sub] = int(parameter_value)
                else: # if float
                    input_parameters[sub] = float(parameter_value)
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
            self.__logger.write('No ini file was specified and the run cannot go further.')
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
        inputParam_dict = iniFile._input_parameters

        # Add a log file
        self.__logger = CE.Logger(output_dir)
        self.__logger.write('FM2PROF version {}\n=============================='.format(__version__))
        self.__logger.write('reading FM and cross-sectional data data')

        # Create an empty list. New cross-sections will be appended to this list. 
        cross_sections = list()

        # Read FM model data
        (time_dependent_data, time_independent_data, edge_data, node_coordinates, cssdata) = FE.read_fm2prof_input(map_file, css_file)
        self.__logger.write('finished reading FM and cross-sectional data data')

        # generate all cross-sections
        for index, name in enumerate(cssdata['id']):
            starttime = datetime.datetime.now()
            self.__logger.write('{} :: cross-section {}'.format(datetime.datetime.strftime(starttime, '%I:%M%p'), name))

            cssindex = cssdata['id'].index(name)
            css = CE.CrossSection(inputParam_dict, 
                                 name=name, 
                                 length=cssdata['length'][cssindex], 
                                 location=cssdata['xy'][cssindex],
                                 branchid=cssdata['branchid'][cssindex],
                                 chainage=cssdata['chainage'][cssindex])
            self.__logger.write('T+ %.2f :: initiated new cross-section %s' % ((datetime.datetime.now()-starttime).total_seconds(), name))

            # Retrieve FM data for cross-section
            fm_data = FE.get_fm2d_data_for_css(css.name,
                                               time_independent_data,
                                               edge_data,
                                               time_dependent_data)

            self.__logger.write('T+ %.2f :: retrieved data for css %s' % ((datetime.datetime.now()-starttime).total_seconds(), name))

            # Build cross-section
            css.build_from_fm(fm_data=fm_data)
            self.__logger.write('T+ %.2f :: cross-section derived, starting correction.....' % (datetime.datetime.now()-starttime).total_seconds())

            # Delta-h correction
            css.calculate_correction()
            self.__logger.write('T+ %.2f :: correction finished' % (datetime.datetime.now()-starttime).total_seconds())

            # Reduce number of points in cross-section
            css.reduce_points(n=20, verbose=False)
            self.__logger.write('T+ %.2f :: simplified cross-section to .. points' % (datetime.datetime.now()-starttime).total_seconds())

            # assign roughness
            css.assign_roughness(fm_data)
            self.__logger.write('T+ %.2f :: computed roughness' % (datetime.datetime.now()-starttime).total_seconds())

            # Append new cross-section to list of cross-sections
            cross_sections.append(css)
            self.__logger.write('cross-section {} generated in {:.2f} seconds'.format(css.name, (datetime.datetime.now()-starttime).total_seconds()))

        # The roughness tables in 1D model require the same discharges on the rows. 
        # This function interpolates to get the roughnesses at the correct discharges
        FE.interpolate_roughness(cross_sections)

        
        chainages = None

        # export all cross-sections
        sobek_export.export_crossSectionLocations(cross_sections, 
            file_path=os.path.join(output_dir, 'CrossSectionLocations.ini'))
        sobek_export.export_geometry(cross_sections, 
                                     file_path=os.path.join(output_dir, 'CrossSectionDefinitions.ini'),
                                     fmt='dflow1d')
        sobek_export.export_geometry(cross_sections, 
                                     file_path=output_dir + '\\geometry.csv',
                                     fmt='sobek3')
        sobek_export.export_geometry(cross_sections, 
                                     file_path=output_dir + '\\geometry_test.csv',
                                     fmt='testformat')
        sobek_export.export_roughness(cross_sections, 
                                      output_dir + '\\roughness.csv',
                                      fmt='sobek3')
        sobek_export.export_roughness(cross_sections, 
                                      output_dir + '\\roughness_test.csv',
                                      fmt='testformat')
        sobek_export.export_volumes(cross_sections, output_dir + '\\volumes.csv')
        self.__logger.write('Exported output files, FM2PROF finished')

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
        self.__logger.write('Saved {0} for {1} plot in {2}.'.format(name, figType, plotLocation))
        
        return
    

# region // Main helpers

def __report_expected_arguments(reason):
    print('main.py -i <map_file> -i <css_file> -i <chainage_file> -o <outputdir>')
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
    """
    Main class, should contain three input arguments and one output.
    Otherwise the execution will end with an error.
    """
    # First try to pars the arguments
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        __report_expected_arguments("Arguments could not be retrieved.")
   
    # Check if number of arguments match the expectation.
    if len(opts) != 4:
        __report_expected_arguments("Not all arguments were given.")
    
    # Check if input parameters are in expected order
    if not __is_input(opts[0]) or not __is_input(opts[1]) or not __is_input(opts[2]):
        __report_expected_arguments("The first three arguments should be input files.\n Given: {0}\n{1}\n{2}\n".format(opts[0], opts[1], opts[2]))
    
    # Check if output parameter is in expected placement
    if not __is_output(opts[3]):
        __report_expected_arguments("The last argument should be the output directory.")


if __name__ == '__main__':
    main(sys.argv[1:])