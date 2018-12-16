import unittest, pytest
import os, sys, getopt, shutil
import TestUtils
from fm2prof import main

class TestMainMethodTest:
    @classmethod
    def setup_class(MainMethodTest):
        """
        Sets up the necessary data for MainMethodTest
        """
        outputtestdir = TestUtils.get_test_data_dir('output_test_main_unit')
        # Start with a clean directory
        if os.path.exists(outputtestdir):
            shutil.rmtree(outputtestdir)
        # just in case
        if os.path.exists(outputtestdir):
            os.rmdir(outputtestdir)

        # Create it (again)
        if not os.path.exists(outputtestdir):
            os.mkdir(outputtestdir)

    @classmethod
    def teardown_class(MainMethodTest):
        """
        Cleans up the directory
        """
        outputtestdir = TestUtils.get_test_data_dir('output_test_main_unit')
        # Remove it.
        if os.path.exists(outputtestdir):
            shutil.rmtree(outputtestdir)
        # just in case
        if os.path.exists(outputtestdir):
            os.rmdir(outputtestdir)

    @pytest.mark.unittest
    def test_main_fails_when_not_giving_arguments(self):
        # 1. Set up test data
        mainArgs = ['']
        
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
    def test_main_fails_when_not_giving_correct_input_arguments(self):
        # 1. Set up test data
        mainArgs = ['-i', 'test1', '-o','test2','--i', 'test3', '--o', 'test5']
        opts, args = getopt.getopt(mainArgs,"hi:o:",["ifile=","ofile="])
        # 2. Set up expectations
        reason = "The first three arguments should be input files.\n Given: {0}\n{1}\n{2}\n".format(opts[0], opts[1], opts[2])
        expectedMssg = "Error: {0}".format(reason)

        # 3. Run test
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main.main(mainArgs)
        
        # 4. Verify expectations
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == expectedMssg

    @pytest.mark.unittest
    def test_main_fails_when_not_giving_correct_output_arguments(self):
        # 1. Set up test data
        mainArgs = ['-i', 'test1', '-i','test2','--i', 'test3', '--i', 'test5']

        # 2. Set up expectations
        reason = "The last argument should be the output directory."
        expectedMssg = "Error: {0}".format(reason)

        # 3. Run test
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main.main(mainArgs)
        
        # 4. Verify expectations
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == expectedMssg

    @pytest.mark.unittest
    def test_main_does_not_SystemExit_when_giving_correct_arguments(self):
        # 1. Set up test data
        outputtestdir = TestUtils.get_test_data_dir('output_test_main_unit')
        mainArgs = ['-i', 'test1', '-i','test2','--i', 'test3', '--o', outputtestdir]
        opts, args = getopt.getopt(mainArgs,"hi:o:",["ifile=","ofile="])

        # 2. Set up expectations
        reasons = ["Not all arguments were given.",
            "The first three arguments should be input files.\n Given: {0}\n{1}\n{2}\n".format(opts[0], opts[1], opts[2]),
            "The last argument should be the output directory."]
        
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
    
    @pytest.mark.unittest
    def test_main_fails_when_giving_correct_arguments_but_non_existent_files(self):
        # 1. Set up test data
        outputtestdir = TestUtils.get_test_data_dir('output_test_main_unit')
        mainArgs = ['-i', 'test1', '-i','test2','--i', 'test3', '--o', outputtestdir]
        opts, args = getopt.getopt(mainArgs,"hi:o:",["ifile=","ofile="])

        # 2. Set up expectations
        reasons = ["Not all arguments were given.",
            "The first three arguments should be input files.\n Given: {0}\n{1}\n{2}\n".format(opts[0], opts[1], opts[2]),
            "The last argument should be the output directory."]
        
        # 3. Run test
        with pytest.raises(Exception) as e_info:
            main.main(mainArgs)

# region // Helpers

# High level acceptance tests, these are the ones who are only meant to generate output files
# for the testers to verify (in Teamcity) whether the runs generate the expected files or not.
def __run_main_with_arguments(map_file, css_file, chainage_file, output_directory):
    pythonCall = "fm2prof\\main.py -i {0} -i {1} -i {2} -o {3}".format(map_file, css_file, chainage_file, output_directory)
    os.system("python {0}".format(pythonCall))

# endregion