
import os
import pytest
from fm2prof import Project
from fm2prof.utils import (
    DeltaresConfig,
    GenerateCrossSectionLocationFile,
    ModelOutputReader,
    Compare1D2D
)
from tests.TestUtils import TestUtils

from datetime import datetime 

_root_output_dir = None


class Test_DeltaresConfig:
    def test_given_input_file_is_read(self):
        # 1. Set up initial test data
        ini_file_path = TestUtils.get_local_test_file(
            "cases/case_02_compound/Model_SOBEK/dimr/dflow1d/NetworkDefinition.ini"
        )

        # 2. Set expectations
        dict_has_keys = ["general", "node", "branch"]

        # 3. Run test
        parsed_dict = DeltaresConfig(ini_file_path)

        # 4. Verify final expectations
        for key in dict_has_keys:
            assert key in parsed_dict.sections


class Test_ModelOutputReader:
    def DUMMY_test_given_1dinput_csv_generated(self):
        # 1. Set up initial test data
        path_1d = TestUtils.get_local_test_file(
            "cases/case_02_compound/Model_SOBEK/dimr/dflow1d/NetworkDefinition.ini"
        )

        # 2. Set expectations

        # 3. Run test
        output = ModelOutputReader.path_flow1d
        output.path_flow1d = path_1d
        output.path_flow2d = path_2d

        output.load_flow1d_data()
        output.get_1d2d_map()
        output.load_flow2d_data()

        # 4. Verify final expectations
        for key in dict_has_keys:
            assert key in parsed_dict.sections


class Test_GenerateCrossSectionLocationFile:
    def test_given_networkdefinitionfile_cssloc_file_is_generated(self):
        # 1. Set up initial test data
        path_1d = TestUtils.get_local_test_file(
            "cases/case_02_compound/Model_SOBEK/dimr/dflow1d/NetworkDefinition.ini"
        )
        output_file = TestUtils.get_local_test_file(
            "cases/case_02_compound/Data/cross_section_locations.xyz"
        )

        # 2. Set Expectations

        # 3. Run test
        GenerateCrossSectionLocationFile(
            networkdefinitionfile=path_1d, crossectionlocationfile=output_file
        )

        # 4. verify
        assert output_file.is_file()


    def test_given_branchrulefile_output_is_generated(self):
        # 1. Set up initial test data
        path_1d = TestUtils.get_local_test_file(
            "cases/case_02_compound/Model_SOBEK/dimr/dflow1d/NetworkDefinition.ini"
        )
        output_file = TestUtils.get_local_test_file(
            "cases/case_02_compound/Data/cross_section_locations_new.xyz"
        )
        if output_file.is_file():
            os.remove(output_file)

        branch_rule_file = TestUtils.get_local_test_file(
            "cases/case_02_compound/Data/branchrules_onlyfirst.ini"
        )
        # 2. Set Expectations

        # 3. Run test
        GenerateCrossSectionLocationFile(
            networkdefinitionfile=path_1d, crossectionlocationfile=output_file,
            branchrulefile=branch_rule_file
        )

        # 4. verify
        assert output_file.is_file()
    

class Test_Compare1D2D:
    def test_when_no_netcdf_but_csv_present_class_initialises(self):
        # 1. Set up initial test data 
        project_config = TestUtils.get_local_test_file('compare1d2d/rijn-j22_6-v1a2/sobek-rijn-j22.ini')
        project = Project(project_config)

        # 2. Set expectations

        # 3. Run test
        try:
            plotter = Compare1D2D(project=project,
                        path_1d=None,
                        path_2d=None,
                        routes=[['BR', "PK", "IJ"], ['BR', 'PK', 'NR', "LE"], ["BR", "WL", "BO"]],
                        start_time=datetime(year=2000, month=1, day=5))
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 4. Verify expectations
        assert isinstance(plotter, Compare1D2D)

    def test_statistics_to_file(self):

        # 1. Set up initial test data 
        project_config = TestUtils.get_local_test_file('compare1d2d/rijn-j22_6-v1a2/sobek-rijn-j22.ini')
        project = Project(project_config)
        plotter = Compare1D2D(project=project,
                    path_1d=None,
                    path_2d=None,
                    routes=[['BR', "PK", "IJ"], ['BR', 'PK', 'NR', "LE"], ["BR", "WL", "BO"]],
                    start_time=datetime(year=2000, month=1, day=5))
        
        # 2. Set expectations
        # this file should exist
        output_file = TestUtils.get_local_test_file('compare1d2d/rijn-j22_6-v1a2/output/error_statistics.csv')

        # 3. Run test
        try:
            plotter.statistics_to_file()
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 4. Verify expectations
        assert output_file.is_file()



    def test_figure_longitudinal(self):

        # 1. Set up initial test data 
        project_config = TestUtils.get_local_test_file('compare1d2d/rijn-j22_6-v1a2/sobek-rijn-j22.ini')
        project = Project(project_config)
        plotter = Compare1D2D(project=project,
                    path_1d=None,
                    path_2d=None,
                    routes=[['BR', "PK", "IJ"], ['BR', 'PK', 'NR', "LE"], ["BR", "WL", "BO"]],
                    start_time=datetime(year=2000, month=1, day=5))
        
        # 2. Set expectations
        # this file should exist
        output_file = TestUtils.get_local_test_file('compare1d2d/rijn-j22_6-v1a2/output/figures/longitudinal/BR-PK-IJ.png')

        # 3. Run test
        try:
            plotter.figure_longitudinal(route=['BR', "PK", "IJ"], stat="last25")
        except Exception as e:
            pytest.fail("No exception expected, but thrown: {}".format(str(e)))

        # 4. Verify expectations
        assert output_file.is_file()