from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fm2prof.imports import ImporterFactory, ModelData
from fm2prof.imports.base import CrossSectionData, FaceGeometry
from fm2prof.imports.csv_elevation import CsvElevationImporter
from fm2prof.imports.dflowfm import DFlowFMImporter
from fm2prof.imports.factory import detect_source
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


class TestCsvElevationImporter:

    CSV_FILE = "cases/case_20_only_elevation/data/mlnbk_triangles.csv"

    @pytest.fixture
    def csv_file(self):
        return TestUtils.get_local_test_file(self.CSV_FILE)

    @pytest.fixture
    def model_data(self, csv_file) -> ModelData:
        return CsvElevationImporter(csv_file).import_data()

    def test_factory_creates_csv_elevation_importer(self, csv_file):
        """ImporterFactory should resolve 'csv_elevation' to CsvElevationImporter."""
        importer = ImporterFactory.create("csv_elevation", csv_file)
        assert isinstance(importer, CsvElevationImporter)

    def test_model_data_is_returned(self, model_data):
        """Import_data should return a ModelData instance."""
        assert isinstance(model_data, ModelData)

    def test_source_is_csv_elevation(self, model_data):
        """Source identifier should be 'csv_elevation'."""
        assert model_data.source == CsvElevationImporter.SOURCE

    def test_face_geometry_is_populated(self, model_data):
        """Geometry should contain face data with positive length."""
        assert model_data.geometry is not None
        assert len(model_data.geometry.x) > 0

    def test_face_geometry_x_y_are_finite_floats(self, model_data):
        """Coordinates x and y should be finite float arrays."""
        assert model_data.geometry.x.dtype == float
        assert model_data.geometry.y.dtype == float
        assert np.all(np.isfinite(model_data.geometry.x))
        assert np.all(np.isfinite(model_data.geometry.y))

    def test_face_geometry_bedlevel_is_populated(self, model_data):
        """Field bedlevel should be a float array of the same length as x."""
        assert len(model_data.geometry.bedlevel) == len(model_data.geometry.x)

    def test_face_geometry_area_is_populated(self, model_data):
        """Field area should be a float array of the same length as x."""
        assert len(model_data.geometry.area) == len(model_data.geometry.x)

    def test_no_edge_data(self, model_data):
        """Edge geometry should not be present for a CSV elevation source."""
        assert model_data.edges is None
        assert not model_data.has_edges

    def test_no_hydraulic_data(self, model_data):
        """Hydraulic data should not be present for a CSV elevation source."""
        assert model_data.hydraulics is None
        assert not model_data.has_hydraulics

    def test_missing_column_raises_value_error(self, tmp_path):
        """A CSV without required columns should raise ValueError."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"POINT_X": [1.0], "POINT_Y": [2.0]}).to_csv(bad_csv, index=False)

        importer = CsvElevationImporter(bad_csv)
        with pytest.raises(ValueError, match="missing required columns"):
            importer.import_data()


class TestDetectSource:

    def test_nc_file_detected_as_dflowfm(self, tmp_path):
        """A .nc file should be detected as 'dflowfm'."""
        nc_file = tmp_path / "FlowFM_map.nc"
        nc_file.touch()
        assert detect_source(nc_file) == "dflowfm"

    def test_csv_file_detected_as_csv_elevation(self, tmp_path):
        """A .csv file should be detected as 'csv_elevation'."""
        csv_file = tmp_path / "elevation.csv"
        csv_file.touch()
        assert detect_source(csv_file) == "csv_elevation"

    def test_extension_matching_is_case_insensitive(self, tmp_path):
        """Extension matching should be case-insensitive."""
        nc_file = tmp_path / "FlowFM_map.NC"
        nc_file.touch()
        assert detect_source(nc_file) == "dflowfm"

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        """An unrecognised extension should raise ValueError."""
        unknown_file = tmp_path / "model.xyz"
        unknown_file.touch()
        with pytest.raises(ValueError, match="Cannot infer source"):
            detect_source(unknown_file)

    def test_accepts_string_path(self, tmp_path):
        """detect_source should accept a plain string path."""
        nc_file = tmp_path / "map.nc"
        nc_file.touch()
        assert detect_source(str(nc_file)) == "dflowfm"
