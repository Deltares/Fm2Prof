import sys, os

def get_test_dir(dirName):
    """
    Returns the desired directory relative to the test data. 
    Avoiding extra code on the tests.
    """
    directory = os.path.dirname(__file__)
    try:
        directory = os.path.join(directory, "test_data\\{0}".format(dirName))
    except:
        print("An error occurred trying to find {0}".format(dirName))
    return directory