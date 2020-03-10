import sys
import os
try:
    from pip import main as pipmain
except:
    from pip._internal import main as pipmain


class TestUtils:

    _name_external = 'external_test_data'
    _name_local = 'test_data'

    @staticmethod
    def install_package(package: str):
        """Installs a package that is normally only used
        by a test configuration.

        Arguments:
            package {str} -- Name of the PIP package.
        """
        pipmain(['install', package])

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
        except Exception:
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

    @staticmethod
    def get_test_dir_output(dir_name: str) -> str:
        """Returns the path to the output test data.
        If it does not exist already it is created.

        Arguments:
            dir_name {str} -- Name of the folder under Output.

        Returns:
            str -- Path to the test output dir.
        """
        output_dir = os.path.join('Output', dir_name)
        test_dir = TestUtils.get_test_dir(output_dir)
        # Create it if it does not exist
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        return test_dir
