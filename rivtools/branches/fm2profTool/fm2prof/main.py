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

from fm2prof import Functions as FE
from fm2prof import Classes as CE
from fm2prof import sobek_export

import os, sys, getopt
# endregion

class Fm2ProfRunner :  
    __logger = None
    
    __showFigures = False
    __saveFigures = False

    __output_dir = None

    def __init__(self, output_dir, showFigures = False, saveFigures = False):
        """
        Initializes the private variables for the Fm2ProfRunner
        """
        self.__logger = CE.Logger(output_dir)
        self.__showFigures = showFigures
        self.__saveFigures = saveFigures
        self.__output_dir = output_dir

    def set_output_directory(self, output_dir):
        """
        Sets the output directory where all generated files from the runs will be stored.
        If this is not set nothing will be saved.
        """
        self.__output_dir = output_dir

    def run_with_files(self, mapFile, cssFile, chainageFile):
        """
        Runs the desired emulation from 2d to 1d given the mapfile and the cross section file.
        """ 
        if not self.__is_output_directory_set():
            return

        # region FILES
        map_file = mapFile
        css_file = cssFile
        chainage_file = chainageFile
        # endregion

        #Just a shortener
        output_dir = self.__output_dir

        cross_sections = list()

        # Read FM model data
        (time_dependent_data, time_independent_data, edge_data, node_coordinates, css_xy, css_names, css_length) = FE.read_fm2prof_input(map_file, css_file)
        self.__logger.write('finished reading FM and cross-sectional data data')

        # generate all cross-sections
        for index, name in enumerate(css_names):
            starttime = datetime.datetime.now()
            self.__logger.write('{} :: cross-section {}'.format(datetime.datetime.strftime(starttime, '%I:%M%p'), name))

            css = CE.CrossSection(name=name, length=css_length[css_names.index(name)], location=css_xy[css_names.index(name)])

            # Retrieve FM data for cross-section
            fm_data = FE.retrieve_for_class(css.name,
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
            self.__logger.write('cross-section {0} generated in {1} seconds'.format(css.name, (datetime.datetime.now()-starttime).total_seconds()))

        # The roughness tables in 1D model require the same discharges on the rows. 
        # This function interpolates to get the roughnesses at the correct discharges
        FE.interpolate_roughness(cross_sections)

        #chainages = FE._read_css_chainages(chainage_file)
        chainages = None

        # export all cross-sections
        sobek_export.geometry_to_csv(cross_sections, chainages, output_dir + '\\geometry.csv')
        sobek_export.roughness_to_csv(cross_sections, chainages, output_dir + '\\roughness.csv')
        sobek_export.volumes_to_csv(cross_sections, chainages, output_dir + '\\volumes.csv')

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
                os.mkdir(self.__output_dir)
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

    map_file = opts[0][1]
    css_file = opts[1][1]
    chainage_file = opts[2][1]
    output_directory = opts[3][1]
    
    # Run with the given arguments
    runner = Fm2ProfRunner(output_directory)
    runner.run_with_files(map_file, css_file, chainage_file)

if __name__ == '__main__':
    main(sys.argv[1:])