import json
import logging
from pathlib import Path

import numpy as np
import pytest

from fm2prof.region_polygon_file import Polygon, MultiPolygon, RegionPolygonFile, SectionPolygonFile
from tests.TestUtils import TestUtils


@pytest.fixture
def polygon_list():
    return [
        Polygon(
            coordinates=[[100, 100], [500, 10], [500, 400], [100, 400], [100, 100]],
            properties={"name": "poly1"},
        ),
        Polygon(
            coordinates=[[300, 300], [900, 300], [900, 400], [300, 400], [300, 300]],
            properties={"name": "poly2"},
        ),
    ]

@pytest.fixture
def test_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "poly1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[100, 100], [500, 10], [500, 400], [100, 400], [100, 100]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "poly2"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[300, 300], [900, 300], [900, 400], [300, 400], [300, 300]],
                },
            },
        ],
    }


def _geojson_file_writer(geojson_dict, file_path) -> None:
    with Path(file_path).open("w") as geojson_file:
        json.dump(geojson_dict, geojson_file, indent=4)


class Test_MultiPolygon:  # noqa: N801

    def test_from_file(self, tmp_path, test_geojson):
        file_path = tmp_path / "polygons.geojson"

        _geojson_file_writer(test_geojson, file_path)
        polygon_file = MultiPolygon(logging.getLogger())
        polygon_file.from_file(file_path=file_path)

        # Verify the polygons were loaded correctly
        assert isinstance(polygon_file.polygons[0],Polygon)

        # Verify the number of polygons loaded
        assert len(polygon_file.polygons) == 2

        # Verify the properties of the polygons
        assert polygon_file.polygons[1].properties.get("name") == "poly2"

    def test_validate_extension(self):
        polygon_file = MultiPolygon(logging.getLogger())
        test_fp = "test.sjon"

        # Test invalid extension raises IOError
        with pytest.raises(IOError, match="Invalid file path extension, should be .json or .geojson."):
            polygon_file._validate_extension(file_path=test_fp)  # noqa: SLF001

        # Test valid extension does not raise
        test_fp = "test.json"
        polygon_file._validate_extension(test_fp)   # noqa: SLF001

    def test_check_overlap(self, polygon_list, mocker):
        polygon_file = MultiPolygon(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list
        mocked_logger = mocker.patch.object(polygon_file, "set_logger_message")
        polygon_file.check_overlap()
        mocked_logger.assert_called_with("poly2 overlaps poly1.", level="warning")

    def test_polygons_property_setter(self, polygon_list):
        """Test that the polygon setter raises errors for invalid input."""
        polygon_file = MultiPolygon(logging.getLogger())
        with pytest.raises(ValueError, match="Polygons must be of type Polygon"):
            polygon_file.polygons = ["test", "case"]

        polygon_list.append(
            Polygon(
                coordinates=[[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]],
                properties={"type": "poly1"},
            ),
        )
        with pytest.raises(ValueError, match="Polygon properties must contain key-word 'name'"):
            polygon_file.polygons = polygon_list
        polygon_list[2].properties.pop("type")
        polygon_list[2].properties["name"] = "poly1"
        with pytest.raises(ValueError, match="Property 'name' must be unique"):
            polygon_file.polygons = polygon_list

        polygon_list.pop()
        polygon_file.polygons = polygon_list
        assert polygon_file.polygons == polygon_list

    def test_get_faces_in_polygon(self, polygon_list, mocker):
        """Test the get_gridpoints_in_polygon method for nodes."""
        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        # Step 2. Instantiate the RegionPolygonFile class
        polygon_file = MultiPolygon(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list
        mocked_logger = mocker.patch.object(polygon_file, "set_logger_message")

        # Step 3. Call the get_gridpoints_in_polygon method
        region_at_points = polygon_file.get_gridpoints_in_polygon(res_file=res_file, dtype="face", property_name="name")

        # Step 4. Verify the output
        assert isinstance(region_at_points, list)
        assert len(region_at_points) == 360
        assert region_at_points[0] == polygon_file.undefined
        assert region_at_points[73] == "poly1"

    def test_get_nodes_in_polygon(self, polygon_list, mocker):
        """Test the get_gridpoints_in_polygon method for faces."""
        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        # Step 2. Instantiate the RegionPolygonFile class
        polygon_file = MultiPolygon(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list
        mocked_logger = mocker.patch.object(polygon_file, "set_logger_message")

        # Step 3. Call the get_gridpoints_in_polygon method
        region_at_points = polygon_file.get_gridpoints_in_polygon(res_file=res_file, dtype="node", property_name="name")

        # Step 4. Verify the output
        assert isinstance(region_at_points, list)
        assert len(region_at_points) == 427
        assert region_at_points[0] == polygon_file.undefined
        assert region_at_points[22] == "poly1"

    def test_get_edges_in_polygon(self, polygon_list, mocker):
        """Test the get_gridpoints_in_polygon method for edges."""
        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        # Step 2. Instantiate the RegionPolygonFile class
        polygon_file = MultiPolygon(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list
        mocked_logger = mocker.patch.object(polygon_file, "set_logger_message")

        # Step 3. Call the get_gridpoints_in_polygon method
        region_at_points = polygon_file.get_gridpoints_in_polygon(res_file=res_file, dtype="edge", property_name="name")

        # Step 4. Verify the output
        assert isinstance(region_at_points, list)
        assert len(region_at_points) == 786
        assert region_at_points[0] == polygon_file.undefined
        assert region_at_points[27] == "poly1"

class Test_RegionPolygonFile:  # noqa: N801

    def test_initialisation(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create RegionPolygonFile instance
        mock_logger = mocker.patch.object(RegionPolygonFile, "set_logger_message")
        region_polygon_file = RegionPolygonFile(region_file_path=file_path, logger=logging.getLogger(__name__))

        # verify logger messages
        assert mock_logger.call_args_list[0][0][0] == "Validating region file"
        assert mock_logger.call_args_list[1][0][0] == "2 regions found"


class Test_SectionPolygonFile:  # noqa: N801

    def test_initialisation(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test_geojson.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create SectionPolygonFile instance with wrong data
        mock_logger = mocker.patch.object(SectionPolygonFile, "set_logger_message")
        with pytest.raises(OSError, match="Section file is not valid"):
            SectionPolygonFile(file_path, logger=logging.getLogger())

        assert mock_logger.call_args_list[1][0][0] == 'Polygon poly1 has no property "section"'
        assert mock_logger.call_args_list[2][0][0] == 'Polygon poly2 has no property "section"'

        # create SectionPolygonFile instance with incorrect data (2)
        test_geojson["features"][0]["properties"]["section"] = "fake section"
        _geojson_file_writer(test_geojson, file_path)
        with pytest.raises(OSError, match="Section file is not valid"):
            SectionPolygonFile(file_path, logger=logging.getLogger())

        assert "fake section is not a recognized section" in [log_cal[0][0] for log_cal in mock_logger.call_args_list]

        # create SectionPolygonFile instance with correct data
        test_geojson["features"][0]["properties"]["section"] = "1"
        test_geojson["features"][1]["properties"]["section"] = "2"
        _geojson_file_writer(test_geojson, file_path)
        section_polygonfile = SectionPolygonFile(file_path, logging.getLogger())
        assert section_polygonfile.sections[0].properties["section"] == "main"
        assert section_polygonfile.sections[1].properties["section"] == "floodplain1"

        assert mock_logger.call_args_list[-1][0][0] == "Section file succesfully validated"
