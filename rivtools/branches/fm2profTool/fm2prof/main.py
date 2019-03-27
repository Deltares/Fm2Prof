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

class Fm2ProfRunner :  
    __logger = None
    
    __showFigures = False
    __saveFigures = False

#    __output_dir = None    
    def __init__(self, IniFile, showFigures = False, saveFigures = False):
        """
        Initializes the private variables for the Fm2ProfRunner
        """
        #self.__logger = CE.Logger(output_dir)
        self.__showFigures = showFigures
        self.__saveFigures = saveFigures
        self.__IniFile = IniFile

    def run(self):
        InputParam_dict = self.read_inifile(self)
        self.run_with_files(self.__mapFile, self.__cssFile, self.__chainageFile, InputParam_dict)

    def read_inifile(self,IniFile):
        """
        Read ini file
        """
        config = configparser.ConfigParser()
        config.read(self.__IniFile)
        
        d = {}
        for section in config.sections():
            d[section] = {}
            for option in config.options(section):
                d[section][option] = config.get(section, option).split('#')[0].strip()
        

        self.ExtractOutputDir(d) # output directory path
        InputParam_dict = self.ExtractInputParameters(d) # dictionary which contains all input parameter values
        self.ExtractInputFiles(d)
        
        return InputParam_dict

    def ExtractInputParameters(self,D):
        """
        Extract InputParameters and convert values either integer or float from string
        """
        ip = D['InputParameters']
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
        for p in D['InputFiles']:
            if 'FM_netCDFile'.lower() in p:
                self.__mapFile = D['InputFiles'][p]
            elif 'CrossSectionLocationFile'.lower() in p:
                self.__cssFile = D['InputFiles'][p]
            elif 'gebiedsvakken'.lower() in p:
                self.__gebiedsvakken = D['InputFiles'][p]
            elif 'SectionFractionFile'.lower() in p:
                self.__sectie = D['InputFiles'][p]
        
        self.__chainageFile = 'tests/external_test_data/case_08_waal/Data/cross_section_chainages.txt' ## it's a dummy; remove it later
    
    def ExtractOutputDir(self, D):
        """
        Extract output directory infomation from the dictionary
        """
        
        if '..' not in D['OutputDirectory']['outputdir']:
            outputdir = os.path.join(os.getcwd(), D['OutputDirectory']['outputdir'])
        else:
            outputdir = D['OutputDirectory']['outputdir'].replace('/','\\')

        casename = D['OutputDirectory']['casename']
        
        if casename == '': # if casename is empty -> use default CaseNameXX
            casename = self.NewCaseName('CaseName',outputdir)
        elif os.path.isdir(os.path.join(outputdir, casename)):
            casename = self.NewCaseName(casename,outputdir)
            
        D['OutputDirectory']['casename'] = casename
        output_dir =  os.path.join(outputdir, D['OutputDirectory']['casename'])
        
        self.__output_dir = output_dir.replace("\\","/")
        
    def NewCaseName(self,casename,outputdir):
        """
        Update casename if already exists
        """
        casenum = 1
        casename_tmp = casename + '{:02d}'.format(casenum)
        while os.path.isdir(os.path.join(outputdir, casename_tmp)):
            casenum += 1
            casename_tmp = 'CaseName' + '{:02d}'.format(casenum)
            
        return casename_tmp
    
    def set_output_directory(self, output_dir):
        """
        Sets the output directory where all generated files from the runs will be stored.
        If this is not set nothing will be saved.
        """
        self.__output_dir = output_dir

    def run_with_files(self, mapFile,cssFile,chainageFile, InputParam_dict):
        """
        Runs the desired emulation from 2d to 1d given the mapfile and the cross section file.
        """ 
        if not self.__is_output_directory_set():
            return

        # region FILES
        map_file = self.__mapFile
        css_file = self.__cssFile
        # chainage_file = chainageFile
        # endregion

        # Just a shortener
        output_dir = self.__output_dir
        # Add a log file
        self.__logger = CE.Logger(output_dir)

        cross_sections = list()

        # Read FM model data
        (time_dependent_data, time_independent_data, edge_data, node_coordinates, css_xy, css_names, css_length) = FE.read_fm2prof_input(map_file, css_file)
        self.__logger.write('finished reading FM and cross-sectional data data')

        # generate all cross-sections
        for index, name in enumerate(css_names):
            starttime = datetime.datetime.now()
            self.__logger.write('{} :: cross-section {}'.format(datetime.datetime.strftime(starttime, '%I:%M%p'), name))

            css = CE.CrossSection(InputParam_dict, name=name, length=css_length[css_names.index(name)], location=css_xy[css_names.index(name)])
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

        #chainages = FE._read_css_chainages(chainage_file)
        chainages = None

        # export all cross-sections
        sobek_export.export_geometry(cross_sections, 
                                     chainages, 
                                     file_path=output_dir + '\\CrossSectionDefinitions.ini',
                                     fmt='dflow1d')
        sobek_export.export_geometry(cross_sections, 
                                     chainages, 
                                     file_path=output_dir + '\\geometry.csv',
                                     fmt='sobek3')
        sobek_export.export_geometry(cross_sections, 
                                     chainages, 
                                     file_path=output_dir + '\\geometry_test.csv',
                                     fmt='testformat')
        sobek_export.export_roughness(cross_sections, 
                                      chainages, 
                                      output_dir + '\\roughness.csv',
                                      fmt='sobek3')
        sobek_export.export_roughness(cross_sections, 
                                      chainages, 
                                      output_dir + '\\roughness_test.csv',
                                      fmt='testformat')
        sobek_export.export_volumes(cross_sections, chainages, output_dir + '\\volumes.csv')
        self.__logger.write('Exported output files, FM2PROF finished')


    def __is_output_directory_set(self):
        """
        Verifies if the output directory has been set and exists or not.
        Returns:
            True - the output_dir is set and exists.
            False - the output_dir is not set or does not exist.
        """
        if self.__output_dir is None:
            print("The output directory must be set before running.")
            return False

        if not os.path.exists(self.__output_dir):
            try:
                os.makedirs(self.__output_dir)
            except:
                
                print("The output directory {0}, could not be found neither created.".format(self.__output_dir))
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