import unittest, pytest
import os, sys, getopt, shutil
import TestUtils
from fm2prof import main

class TestMainMethod_UnitTest:
    @classmethod
    def setup_class(MainMethod_UnitTest):
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
    def teardown_class(MainMethod_UnitTest):
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
        mainArgs = ['-o', 'test1']
        opts, args = getopt.getopt(mainArgs,"hi:o:",["ifile=","ofile="])
        # 2. Set up expectations
        reason = "The first argument should be an input file.\n Given: {0}\n".format(opts[0])
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
    
    @pytest.mark.integrationtest
    def test_main_when_non_existent_input_file_then_raises_io_exception(self):
        # 1. Set up test data
        file_path = 'test1'
        mainArgs = ['-i', file_path]

        # 2. Set up expectations
        reason = 'The given file path {} could not be found.'.format(file_path)
        
        # 3. Run test
        with pytest.raises(IOError) as e_info:
            main.main(mainArgs)
        
        # 4. Verify final expectations
        exception_message = str(e_info.value)
        assert exception_message == reason, 'Expected exception message {}, retrieved {}'.format(reason, exception_message)
    
    @pytest.mark.integrationtest
    def test_main_when_existent_input_file_then_does_not_raise_io_exception(self):
        # 1. Set up test data
        test_dir = TestUtils.get_test_data_dir('main_test_data')
        file_name = 'test_ini_file.ini'
        file_path = os.path.join(test_dir, file_name)
        mainArgs = ['-i', file_path]

        # 2. Set up expectations
        assert os.path.exists(file_path)
        reason = 'The given file path {} could not be found.'.format(file_path)
        
        # 3. Run test
        try:
            main.main(mainArgs)
        except IOError:
            pytest.fail('Unexpected IOError exception.')
    
# region // Helpers

# High level acceptance tests, these are the ones who are only meant to generate output files
# for the testers to verify (in Teamcity) whether the runs generate the expected files or not.
def __run_main_with_arguments(ini_file):
    pythonCall = "fm2prof\\main.py -i {0}".format(ini_file)
    os.system("python {0}".format(pythonCall))

# endregion