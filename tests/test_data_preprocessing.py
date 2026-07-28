import numpy as np

from fm2prof.data_preprocessing import build_model_data
from fm2prof.ini_file import InputFiles
from tests.TestUtils import TestUtils


class TestClassification:

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
