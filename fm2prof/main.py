# -*- coding: utf-8 -*-
"""
Top-level file to run fm2prof
"""


import getopt

# Import from standard library
import sys

# Import from package
from fm2prof import Fm2ProfRunner

# Import from dependencies
# None




# region // Main helpers
def __report_expected_arguments(reason):
    print("main.py -i <ini_file>")
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
    """Main class, should contain three input arguments and one output.
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
        err_mssg = (
            ""
            + "The first argument should be an input file.\n"
            + "Given: {}\n".format(opts[0])
        )
        __report_expected_arguments(err_mssg)

    # Run Fm2Prof with given arguments
    ini_file_path = opts[0][1]
    runner = Fm2ProfRunner.Fm2ProfRunner(ini_file_path)
    runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
