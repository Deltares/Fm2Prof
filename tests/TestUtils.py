import sys, os

def get_test_data_dir(dirName):
    """
    Returns the desired directory relative to the test data. 
    Avoiding extra code on the tests.
    """
    directory = os.path.dirname(__file__)
    try:
        directory = os.path.join(directory, "test_data\\{0}\\".format(dirName))
    except:
        print("An error occurred trying to find {0}".format(dirName))
    return directory

def get_external_test_data_dir(dirName):
    """
    Returns the desired directory relative to the test external data. 
    Avoiding extra code on the tests.
    """
    directory = os.path.dirname(__file__)
    try:
        directory = os.path.join(directory, "external_test_data\\{0}\\".format(dirName))
    except:
        print("An error occurred trying to find {0}".format(dirName))
    return directory