import getopt
import os
import shutil
import sys
import unittest

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

    @pytest.mark.unittest
    def test_when_incorrect_args_then_systemexit_risen_with_expected_message(self):
        # 1. Set up test data
        mainArgs = [""]

        # 2. Set up expectations
        reason = "Not all arguments were given."
        expectedMssg = "Error: {0}".format(reason)

        # 3. Run test
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main.main(mainArgs)

        # 4. Verify expectations
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == expectedMssg

    @pytest.mark.unittest
    def test_when_incorrect_input_args_systemexit_risen_with_expected_message(self):
        # 1. Set up test data
        mainArgs = ["-o", "test1"]
        opts, args = getopt.getopt(mainArgs, "hi:o:", ["ifile=", "ofile="])
        # 2. Set up expectations
        reason = "The first argument should be an input file.\n" + "Given: {}\n".format(
            opts[0]
        )
        expectedMssg = "Error: {0}".format(reason)

        # 3. Run test
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main.main(mainArgs)

        # 4. Verify expectations
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == expectedMssg

    @pytest.mark.unittest
    def test_when_giving_correct_arguments_then_does_not_raise_systemexit(self):
        # 1. Set up test data
        test_dir = "output_test_main_unit"
        outputtestdir = TestUtils.get_local_test_data_dir(test_dir)
        mainArgs = ["-i", "test1", "-i", "test2", "--i", "test3", "--o", outputtestdir]
        opts, args = getopt.getopt(mainArgs, "hi:o:", ["ifile=", "ofile="])

        # 2. Set up expectations
        reasons = [
            "Not all arguments were given.",
            "The first three arguments should be input files.\n"
            + " Given: {}\n{}\n{}\n".format(opts[0], opts[1], opts[2]),
            "The last argument should be the output directory.",
        ]

        # 3. Run test
        try:
            with pytest.raises(SystemExit) as pytest_wrapped_e:
                main.main(mainArgs)
            # 4. Verify expectations (it should actually not get here)
            assert pytest_wrapped_e.type == SystemExit
            for reason in reasons:
                expectedMssg = "Error: {0}".format(reason)
                assert pytest_wrapped_e.value.code != expectedMssg
        except:
            pass

    @pytest.mark.integrationtest
    def test_when_giving_non_existent_input_file_then_raises_io_exception(self):
        # 1. Set up test data
        file_path = "test1"
        mainArgs = ["-i", file_path]

        # 2. Set up expectations
        reason = "The given file path {} could not be found.".format(file_path)

        # 3. Run test
        with pytest.raises(IOError) as e_info:
            main.main(mainArgs)

        # 4. Verify final expectations
        exception_message = str(e_info.value)
        assert (
            exception_message == reason
        ), "" + "Expected exception message {}, retrieved {}".format(
            reason, exception_message
        )

    @pytest.mark.integrationtest
    def ARCHIVED_test_when_giving_existent_empty_input_file_then_does_not_raise_io_exception(
        self,
    ):
        # 1. Set up test data
        test_dir = TestUtils.get_local_test_data_dir("main_test_data")
        file_name = "test_ini_file.ini"
        file_path = os.path.join(test_dir, file_name)
        mainArgs = ["-i", file_path]

        # 2. Set up expectations
        assert os.path.exists(file_path)
        reason = "The given file path {} could not be found.".format(file_path)

        # 3. Run test
        try:
            main.main(mainArgs)
        except:
            pytest.fail("Unexpected exception.")
