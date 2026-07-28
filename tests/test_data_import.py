from pathlib import Path

import numpy as np

from fm2prof.imports import ModelData
from fm2prof.imports.base import CrossSectionData, FaceGeometry
from fm2prof.imports.dflowfm import DFlowFMImporter
from tests.TestUtils import TestUtils, skipwhenexternalsmissing


class TestFMDataImporter:
    @skipwhenexternalsmissing
    def test_when_map_file_without_czu_no_exception(self):
        # 1. Set up test data
        test_map = Path(TestUtils.get_local_test_data_dir("main_test_data")).joinpath("fm_map.nc")
        assert test_map.is_file()

        # 2. Set initial expectations
        importer = DFlowFMImporter(test_map)
        return_value = importer.import_data()

        # 3. Verify final expectations
        assert return_value is not None

    def test_initialisation_with_map_file(self):
        test_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        fmdata = DFlowFMImporter(test_file)

        assert fmdata is not None
        assert fmdata.file_path == test_file

    def test_get_variable(self):
        test_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        fmdata = DFlowFMImporter(test_file)

        var_data = fmdata.get_variable("mesh2d_face_x")

        assert var_data is not None
        assert isinstance(var_data, np.ndarray)
        assert len(var_data) == 360
        assert var_data[0] == 25.0
        assert var_data[-1] == 2975.0

class TestFmModelData:
    def _make_geometry(self, n: int = 3) -> FaceGeometry:
        return FaceGeometry(
            x=np.zeros(n), y=np.zeros(n), area=np.ones(n), bedlevel=np.zeros(n),
            section=np.array(["main"] * n, dtype="U99"),
            region=np.array([""] * n, dtype="U99"),
            sclass=np.array([""] * n, dtype="U99"),
            islake=np.zeros(n, dtype=bool),
        )

    def test_when_given_expected_arguments_then_object_is_created(self):
        # 1. Set up test data
        geometry = self._make_geometry()
        cross_sections = []

        # 2. Run test
        return_model_data = ModelData(
            geometry=geometry,
            cross_sections=cross_sections,
            source="dflowfm",
        )

        # 3. Verify final expectations
        assert return_model_data is not None
        assert return_model_data.geometry is geometry
        assert return_model_data.cross_sections == []
        assert return_model_data.source == "dflowfm"
        assert return_model_data.edges is None
        assert return_model_data.hydraulics is None

    def test_when_given_cross_sections_then_cross_sections_are_set(self):
        # 1. Set up test data
        css = CrossSectionData(name="css_001", length=100.0, location=(1.0, 2.0), branch_id="branch_a", offset=50.0)

        # 2. Run test
        model_data = ModelData(geometry=self._make_geometry(), cross_sections=[css], source="dflowfm")

        # 3. Verify
        assert len(model_data.cross_sections) == 1
        assert model_data.cross_sections[0].name == "css_001"

    def test_has_edges_false_when_not_provided(self):
        model_data = ModelData(geometry=self._make_geometry(), cross_sections=[], source="dflowfm")
        assert model_data.has_edges is False

    def test_has_hydraulics_false_when_not_provided(self):
        model_data = ModelData(geometry=self._make_geometry(), cross_sections=[], source="dflowfm")
        assert model_data.has_hydraulics is False
