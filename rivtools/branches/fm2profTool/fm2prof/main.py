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
import sys
import getopt

from fm2prof import Fm2ProfRunner

pd.options.mode.chained_assignment = None  # default='warn'

# endregion

__version__ = 1.1
__revision__ = 2

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
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        __report_expected_arguments("Arguments could not be retrieved.")

    # Check if number of arguments match the expectation.
    if len(opts) != 1:
        __report_expected_arguments("Not all arguments were given.")

    # Check if input parameters are in expected order
    if not __is_input(opts[0]):
        err_mssg = '' + \
            'The first argument should be an input file.\n' + \
            'Given: {}\n'.format(opts[0])
        __report_expected_arguments(err_mssg)

    # Run Fm2Prof with given arguments
    ini_file_path = opts[0][1]
    runner = Fm2ProfRunner.Fm2ProfRunner(ini_file_path, __version__)
    runner.run()


if __name__ == '__main__':
    main(sys.argv[1:])
