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

import os
# endregion

def runfile(dataDir, mapFile, cssFile, chainageFile):
    """
    Runs the desired emulation from 2d to 1d given the mapfile and the cross section file.
    TODO: datadir should be optional ?(to know where to direct the output)
    """ 
    # region FILES
    directory = dataDir
    map_file = mapFile
    css_file = cssFile
    chainage_file = chainageFile
    # endregion

    cross_sections = list()

    # Read FM model data
    (time_dependent_data, time_independent_data, edge_data, node_coordinates, css_xy, css_names, css_length) = FE.read_fm2prof_input(map_file, css_file)
    print('finished reading FM and cross-sectional data data')

    # Initiate classes
    logger = CE.LoggerClass()

    # generate all cross-sections
    for index, name in enumerate(css_names):
        starttime = datetime.datetime.now()
        #logger.write('%s :: cross-section %s' % (datetime.datetime.strftime(starttime, '%I:%M%p'), name))

        css = CE.CrossSection(name=name, length=css_length[css_names.index(name)], location=css_xy[css_names.index(name)])

        # Retrieve FM data for cross-section
        fm_data = FE.retrieve_for_class(css.name,
                                        time_independent_data,
                                        edge_data,
                                        time_dependent_data)

        #logger.write('T+ %.2f :: retrieved data for css %s' % ((datetime.datetime.now()-starttime).total_seconds(), name))

        # Build cross-section
        css.build_from_fm(fm_data=fm_data)
        #logger.write('T+ %.2f :: cross-section derived, starting correction.....' % (datetime.datetime.now()-starttime).total_seconds())

        # Delta-h correction
        css.calculate_correction()
        #css._plot_volume(relative_error=True)
        #logger.write('T+ %.2f :: correction finished' % (datetime.datetime.now()-starttime).total_seconds())

        # Reduce number of points in cross-section
        css.reduce_points(n=20, verbose=False)

        # assign roughness
        css.assign_roughness(fm_data)
        #css._plot_roughness(fm_data, True)

        #css._plot_zw()
        #plt.show()

        logger.write('cross-section {0} generated in {1} seconds'.format(css.name, (datetime.datetime.now()-starttime).total_seconds()))
        cross_sections.append(css)

    FE.interpolate_roughness(cross_sections)

    #chainages = FE._read_css_chainages(chainage_file)
    chainages = None

    # export all cross-sections
    sobek_export.geometry_to_csv(cross_sections, chainages, directory + 'Output\\geometry.csv')
    sobek_export.roughness_to_csv(cross_sections, chainages, directory + 'Output\\roughness.csv')

def main():
    # hardcoded for now, it should be introduced through command line (or call directly to runfile method)
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, '..\\tests\\test_data\\case_01_rectangle\\')
    map_file = directory + 'Data\\FM\\50x25_mesh\\FlowFM_fm2prof_map.nc'
    css_file = directory + 'Data\\cross_section_locations.xyz'
    chainage_file = directory + 'Data\\cross_section_chainages.txt'
    print("Running with hardcoded path from: {0}".format(directory))
    runfile(directory, map_file, css_file, chainage_file)

main()