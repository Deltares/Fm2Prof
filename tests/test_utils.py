import numbers
import os
import shutil
import sys
import unittest

import pytest

from fm2prof.utils import (
    DeltaresConfig,
    GenerateCrossSectionLocationFile,
    ModelOutputReader,
)
from tests.TestUtils import TestUtils

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
    
