import numpy as np
import pytest

from fm2prof.data_preprocessing import build_model_data
from fm2prof.ini_file import InputFiles
from tests.TestUtils import TestUtils


class TestClassification:

    @pytest.fixture(autouse=True)
    def clear_polygon_caches(self):
        """Delete any cached region/section classification files before each test.

        Cache files are placed next to the map file and have the pattern
        ``<mapfile>.region_cache.json`` and ``<mapfile>.section_cache.json``.
        Removing them ensures each test run performs a fresh classification.
        """
        cache_dir = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput")
        for cache_file in cache_dir.glob("*_cache.json"):
            cache_file.unlink(missing_ok=True)

    def test_region_polygon_assigns_all_faces_to_poly1(self):
        """All 2D faces should be assigned to region 'poly1' for case_02_compound_with_region."""
        # 1. Set up input paths from the case config
        case_dir = TestUtils.get_local_test_file("cases/case_02_compound/Data")
        input_files = InputFiles(
            map_file=case_dir / "2DModelOutput" / "FlowFM_map.nc",
            css_file=case_dir / "cross_section_locations.xyz",
            region_file=case_dir / "region_polygon.geojson",
            section_file=None,
        )

        # 2. Build model data
        model_data = build_model_data(input_files)

        # 3. Verify all faces are assigned to region 'poly1'
        assert model_data.geometry.region is not None
        assert len(model_data.geometry.region) > 0
        assert np.all(model_data.geometry.region == "poly1"), (
            f"Expected all faces to have region 'poly1', "
            f"but found: {np.unique(model_data.geometry.region)}"
        )


# Mirror the acceptance test cases as raw path dicts so that TestUtils can
# resolve them at test-run time rather than at import time.
_PREPROCESSING_CASES = [
    pytest.param(
        {
            "map_file":    "cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc",
            "css_file":    "cases/case_02_compound/Data/cross_section_locations.xyz",
            "region_file": None,
            "section_file": None,
        },
        id="case_02_compound",
    ),
    pytest.param(
        {
            "map_file":    "cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc",
            "css_file":    "cases/case_02_compound/Data/cross_section_locations.xyz",
            "region_file": "cases/case_02_compound/Data/region_polygon.geojson",
            "section_file": None,
        },
        id="case_02_compound_with_region",
    ),
    pytest.param(
        {
            "map_file":    "cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc",
            "css_file":    "cases/case_02_compound/Data/cross_section_locations.xyz",
            "region_file": "cases/case_02_compound/Data/region_polygon.geojson",
            "section_file": "cases/case_02_compound/Data/section_polygon.geojson",
        },
        id="case_02_compound_with_region_and_section",
    ),
    pytest.param(
        {
            "map_file":    "cases/case_20_only_elevation/data/mlnbk_triangles.csv",
            "css_file":    "cases/case_20_only_elevation/model/CrossSectionLocations.xyz",
            "region_file": None,
            "section_file": None,
        },
        id="case_20_elevation_only",
        marks=[pytest.mark.xfail(reason="CSV elevation source not yet fully supported in build_model_data")],
    ),
]


class TestBuildModelData:

    @pytest.fixture(autouse=True)
    def clear_polygon_caches(self):
        """Delete cached region/section files before each test."""
        cache_dir = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput")
        for cache_file in cache_dir.glob("*_cache.json"):
            cache_file.unlink(missing_ok=True)

    @pytest.mark.parametrize("case", _PREPROCESSING_CASES)
    def test_returns_model_data_instance(self, case):
        """build_model_data should return a ModelData instance for every supported case."""
        from fm2prof.imports.base import ModelData

        input_files = InputFiles(
            map_file=TestUtils.get_local_test_file(case["map_file"]),
            css_file=TestUtils.get_local_test_file(case["css_file"]),
            region_file=(
                TestUtils.get_local_test_file(case["region_file"])
                if case["region_file"] else None
            ),
            section_file=(
                TestUtils.get_local_test_file(case["section_file"])
                if case["section_file"] else None
            ),
        )

        result = build_model_data(input_files)

        assert isinstance(result, ModelData)
        assert result.geometry is not None
        assert len(result.geometry.x) > 0
