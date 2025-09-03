from pathlib import Path

import numpy as np
import pytest

from fm2prof.data_import import FMDataImporter, FmModelData
from tests.TestUtils import TestUtils, skipwhenexternalsmissing


class Test_FMDataImporter:
    @skipwhenexternalsmissing
    def test_when_map_file_without_czu_no_exception(self):
        # 1. Set up test data
        test_map = Path(TestUtils.get_local_test_data_dir("main_test_data")).joinpath("fm_map.nc")
        assert test_map.is_file()

        # 2. Set initial expectations
        return_value = FMDataImporter().import_dflow2d(test_map)

        # 3. Verify final expectations
        assert return_value is not None

    def test_initialisation_with_map_file(self):
        test_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        fmdata = FMDataImporter(test_file)

        assert fmdata is not None
        assert fmdata.file_path == test_file

    def test_get_variable(self):
        test_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        fmdata = FMDataImporter(test_file)

        var_data = fmdata.get_variable("mesh2d_face_x")

        assert var_data is not None
        assert isinstance(var_data, np.ndarray)
        assert len(var_data) == 360
        assert var_data[0] == 25.0
        assert var_data[-1] == 2975.0

class Test_FmModelData:
    def test_when_given_expected_arguments_then_object_is_created(self):
        # 1. Set up test data
        time_dependent_data = "arg1"
        time_independent_data = "arg2"
        edge_data = "arg3"
        node_coordinates = "arg4"
        css_data = "arg5"
        return_fm_model_data = None

        # 2. Run test
        return_fm_model_data = FmModelData(
            time_dependent_data=time_dependent_data,
            time_independent_data=time_independent_data,
            edge_data=edge_data,
            node_coordinates=node_coordinates,
            css_data_dictionary=css_data,
        )

        # 4. Verify final expectations
        assert return_fm_model_data is not None
        assert return_fm_model_data.time_dependent_data == time_dependent_data
        assert return_fm_model_data.time_independent_data == time_independent_data
        assert return_fm_model_data.edge_data == edge_data
        assert return_fm_model_data.node_coordinates == node_coordinates
        assert return_fm_model_data.css_data_list == []

    def test_when_given_data_dictionary_then_css_data_list_is_set(self):
        # 1. Set up test data
        time_dependent_data = "arg1"
        time_independent_data = "arg2"
        edge_data = "arg3"
        node_coordinates = "arg4"
        dummy_key = "dummyKey"
        dummy_values = [0, 1]
        css_data_dict = {
            dummy_key: dummy_values,
        }

        return_fm_model_data = None

        # 2. Set expectations
        expected_css_data_list = [{dummy_key: 0}, {dummy_key: 1}]

        #  3. Run test
        return_fm_model_data = FmModelData(
            time_dependent_data=time_dependent_data,
            time_independent_data=time_independent_data,
            edge_data=edge_data,
            node_coordinates=node_coordinates,
            css_data_dictionary=css_data_dict,
        )

        # 4. Verify final expectations
        assert return_fm_model_data is not None
        assert return_fm_model_data.css_data_list != css_data_dict
        assert return_fm_model_data.css_data_list == expected_css_data_list


class Test_get_ordered_css_list:
    def test_when_given_dictionary_then_returns_list(self):
        # 1. Set up test_data
        return_list = None
        dummy_key = "dummyKey"
        dummy_values = [0, 1]
        test_dict = {
            dummy_key: dummy_values,
        }

        # 2. Run test
        expected_list = [{dummy_key: 0}, {dummy_key: 1}]

        # 3. Run test
        return_list = FmModelData.get_ordered_css_list(test_dict)

        # 4. Verify final expectations
        assert return_list is not None
        assert return_list == expected_list, (
            "" + "Expected return value {},".format(expected_list) + " but return {} instead.".format(return_list)
        )

    @pytest.mark.parametrize("test_dict", [(""), (None), ({})])
    def test_when_given_unexpected_value_then_returns_empty_list(self, test_dict):
        # 1. Set up test_data
        return_list = None

        # 2. Run test
        expected_list = []

        # 3. Run test
        return_list = FmModelData.get_ordered_css_list(test_dict)

        # 4. Verify final expectations
        assert return_list is not None
        assert return_list == expected_list, (
            "" + "Expected return value {},".format(expected_list) + " but return {} instead.".format(return_list)
        )
