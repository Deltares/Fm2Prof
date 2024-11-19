import getopt
import os
import shutil
import re

import pytest

from fm2prof import main
from tests.TestUtils import TestUtils


class Test_Main:
    @classmethod
    def setup_class(Main):
        """
        Sets up the necessary data for MainMethodTest
        """
        test_dir = "output_test_main_unit"
        outputtestdir = TestUtils.get_local_test_data_dir(test_dir)
        # Start with a clean directory
        if os.path.exists(outputtestdir):
            shutil.rmtree(outputtestdir)
        # just in case
        if os.path.exists(outputtestdir):
            os.rmdir(outputtestdir)

        # Create it (again)
        if not os.path.exists(outputtestdir):
            os.makedirs(outputtestdir)

    @classmethod
    def teardown_class(Main):
        """
        Cleans up the directory
        """
        test_dir = "output_test_main_unit"
        outputtestdir = TestUtils.get_local_test_data_dir(test_dir)
        # Remove it.
        if os.path.exists(outputtestdir):
            shutil.rmtree(outputtestdir)
        # just in case
        if os.path.exists(outputtestdir):
            os.rmdir(outputtestdir)

    def test_when_incorrect_args_then_systemexit_raised_with_expected_message(self):
        # 1. Set up test data
        mainArgs = [""]

        # 2. Set up expectations
        expectedMssg = "Error: Not all arguments were given."

        # 3. Run test
        with pytest.raises(SystemExit, match=expectedMssg):
            main.main(mainArgs)


    def test_when_incorrect_input_args_systemexit_raised_with_expected_message(self):
        # 1. Set up test data
        mainArgs = ["-o", "test1"]
        opts, args = getopt.getopt(mainArgs, "hi:o:", ["ifile=", "ofile="])
        # 2. Set up expectations
        reason = "The first argument should be an input file.\n" + "Given: {}\n".format(
            opts[0]
        )
        expectedMssg = "Error: {0}".format(reason)

        # 3. Run test
        with pytest.raises(SystemExit, match=re.escape(expectedMssg)):
            main.main(mainArgs)

       
    def test_when_giving_correct_arguments_then_does_not_raise_systemexit(self):
        # 1. Set up test data
        test_dir = "output_test_main_unit"
        outputtestdir = TestUtils.get_local_test_data_dir(test_dir)
        mainArgs = ["-i", "test1", "-i", "test2", "--i", "test3", "--o", outputtestdir]
        opts, args = getopt.getopt(mainArgs, "hi:o:", ["ifile=", "ofile="])

        # 2. Run test
        with pytest.raises(SystemExit, match="Not all arguments were given."):
            main.main(mainArgs)
        
       

