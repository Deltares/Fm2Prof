import sys
import os


class TestUtils:

    _name_external = 'external_test_data'
    _name_local = 'test_data'

    @staticmethod
    def get_local_test_data_dir(dir_name: str):
        """
        Returns the desired directory relative to the test data.
        Avoiding extra code on the tests.
        """
        directory = TestUtils.get_test_data_dir(
            dir_name, TestUtils._name_local)
        return directory

    @staticmethod
    def get_external_test_data_dir(dir_name: str):
        """
        Returns the desired directory relative to the test external data.
        Avoiding extra code on the tests.
        """
        directory = TestUtils.get_test_data_dir(
            dir_name, TestUtils._name_external)
        return directory

    @staticmethod
    def get_test_data_dir(dir_name: str, test_data_name: str):
        """
        Returns the desired directory relative to the test external data.
        Avoiding extra code on the tests.
        """
        test_dir = os.path.dirname(__file__)
        try:
            dir_path = '{}\\{}\\'.format(test_data_name, dir_name)
            test_dir = os.path.join(test_dir, dir_path)
        except:
            print("An error occurred trying to find {}".format(dir_name))
        return test_dir

    @staticmethod
    def get_test_dir(dir_name: str):
        """Returns the desired directory inside the Tests folder

        Arguments:
            dir_name {str} -- Target directory.

        Returns:
            {str} -- Path to the target directory.
        """
        test_dir = os.path.dirname(__file__)
        dir_path = os.path.join(test_dir, dir_name)
        return dir_path
