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

import os, sys, getopt
# endregion

class IniFile:
    __logger = None
    __filePath = None
    # region Public parameters
    _map_file = None
    _css_file = None
    _output_dir = None
    _chainageFile = None
    _gebiedsVakken = None
    _sectie = None
    _inputParam_dict = None
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
        
        d = {}
        for section in config.sections():
            d[section] = {}
            for option in config.options(section):
                d[section][option] = config.get(section, option).split('#')[0].strip()
        

        self.ExtractOutputDir(d) # output directory path
        self._inputParam_dict = self.ExtractInputParameters(d) # dictionary which contains all input parameter values
        self.ExtractInputFiles(d)

    def ExtractInputParameters(self,D):
        """
        Extract InputParameters and convert values either integer or float from string
        """
        inputParametersKey = 'InputParameters'
        ip = D[inputParametersKey]
        for sub in ip:
            try:
                if abs(int(ip[sub])-float(ip[sub])) < 1e-6: # if integer
                    ip[sub] = int(ip[sub])
                else: # if float
                    ip[sub] = float(ip[sub])
            except ValueError:
                ip[sub] = float(ip[sub])
        return ip
        
    def ExtractInputFiles(self,D):
        """
        Extract input file information from the dictionary
        """
        inputFilesKey = 'InputFiles'
        for p in D[inputFilesKey]:
            if 'FM_netCDFile'.lower() in p:
                self._map_file = D[inputFilesKey][p]
            elif 'CrossSectionLocationFile'.lower() in p:
                self._css_file = D[inputFilesKey][p]
            elif 'gebiedsvakken'.lower() in p:
                self._gebiedsvakken = D[inputFilesKey][p]
            elif 'SectionFractionFile'.lower() in p:
                self._sectie = D[inputFilesKey][p]
    
    def ExtractOutputDir(self, D):
        """
        Extract output directory infomation from the dictionary
        """
        outputDirectoryKey = 'OutputDirectory'
        outputDirKey = 'outputdir'
        casenameKey = 'casename'
        if '..' not in D[outputDirectoryKey][outputDirKey]:
            outputdir = os.path.join(os.getcwd(), D[outputDirectoryKey][outputDirKey])
        else:
            outputdir = D[outputDirectoryKey][outputDirKey].replace('/','\\')

        casename = D[outputDirectoryKey][casenameKey]
        
        if casename == '': # if casename is empty -> use default CaseNameXX
            casename = self.NewCaseName('CaseName',outputdir)
        elif os.path.isdir(os.path.join(outputdir, casename)):
            casename = self.NewCaseName(casename,outputdir)
            
        D[outputDirectoryKey][casenameKey] = casename
        output_dir =  os.path.join(outputdir, D[outputDirectoryKey][casenameKey])
        
        self._output_dir = output_dir.replace("\\","/")

    def NewCaseName(self, casename : str, outputdir : str):
        """
        Update casename if already exists
        """
        casenum = 1
        casename_tmp = casename + '{:02d}'.format(casenum)
        while os.path.isdir(os.path.join(outputdir, casename_tmp)):
            casenum += 1
            casename_tmp = 'CaseName' + '{:02d}'.format(casenum)
            
        return casename_tmp
  

class Fm2ProfRunner :  
    __logger = None    
    __iniFile = None
    __showFigures = False
    __saveFigures = False

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
      
    def set_output_directory(self, output_dir):
        """
        Sets the output directory where all generated files from the runs will be stored.
        If this is not set nothing will be saved.
        """
        self.__output_dir = output_dir

    def run_inifile(self, iniFile : IniFile):
        """Runs the desired emulation from 2d to 1d given the mapfile and the cross section file.
        
        Arguments:
            iniFile {IniFile} -- Object containing all the information needed to execute the program
        """
        if not self.__is_output_directory_set(iniFile):
            return

        # shorter local variables
        map_file = iniFile._map_file
        css_file = iniFile._css_file
        output_dir = iniFile._output_dir
        inputParam_dict = iniFile._inputParam_dict

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

    def __generate_output(self, output_directory, fig, figType, name):
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