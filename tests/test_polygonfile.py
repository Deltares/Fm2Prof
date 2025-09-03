import json
import logging
from pathlib import Path

import pytest

from fm2prof.polygon_file import MultiPolygon, Polygon, PolygonError, RegionPolygon, SectionPolygon
from tests.TestUtils import TestUtils


@pytest.fixture
def polygon_list():
    return [
        Polygon(
            coordinates=[[100, 100], [500, 10], [500, 400], [100, 400], [100, 100]],
            properties={"region": "poly1", "name": "poly1"},
        ),
        Polygon(
            coordinates=[[300, 300], [900, 300], [900, 400], [300, 400], [300, 300]],
            properties={"region": "poly2", "name": "poly2"},
        )
    ]

@pytest.fixture
def test_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "poly1", "region": "poly1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[100, 100], [500, 100], [500, 400], [100, 400], [100, 100]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "poly2", "region": "poly2"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[300, 300], [900, 300], [900, 400], [300, 400], [300, 300]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "multipoly1", "region": "multipoly1"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [[[[300, 300], [900, 300], [900, 400], [300, 400], [300, 300]]]],
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
        assert isinstance(polygon_file.polygons[0], Polygon)

        # Verify the number of polygons loaded
        assert len(polygon_file.polygons) == 3

        # Verify the properties of the polygons
        assert polygon_file.polygons[1].properties.get("name") == "poly2"
        assert polygon_file.polygons[2].properties.get("name") == "multipoly1"

    def test_from_file_with_invalid_json(self, tmp_path, test_geojson):
        file_path = tmp_path / "polygons.geojson"

        _geojson_file_writer(test_geojson, file_path)
        polygon_file = MultiPolygon(logging.getLogger())

        # Corrupt the file by writing invalid JSON
        with file_path.open("w") as f:
            f.write("{invalid_json: true,}")

        with pytest.raises(PolygonError, match="Error decoding JSON from"):
            polygon_file.from_file(file_path=file_path)

    def test_from_file_invalid(self, tmp_path):
        file_path = tmp_path / "polygons.geojson"

        # Write invalid geojson (missing 'features' key)
        invalid_geojson = {"type": "FeatureCollection"}
        _geojson_file_writer(invalid_geojson, file_path)

        polygon_file = MultiPolygon(logging.getLogger())
        with pytest.raises(PolygonError, match="Polygon file has no features"):
            polygon_file.from_file(file_path=file_path)


    def test_from_file_nonexistent(self):
        polygon_file = MultiPolygon(logging.getLogger())
        with pytest.raises(FileNotFoundError, match="Polygon file does not exist"):
            polygon_file.from_file(file_path="non_existent_file.geojson")

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
        faces_in_polygon = polygon_file.meshkernel_inpolygon(res_file=res_file, dtype="face", property_name="region")

        # Step 4. Verify the output
        assert isinstance(faces_in_polygon, list)
        assert len(faces_in_polygon) == 360
        assert faces_in_polygon[0] == polygon_file.undefined
        assert faces_in_polygon[73] == "poly1"

    def test_get_nodes_in_polygon(self, polygon_list, mocker):
        """Test the get_gridpoints_in_polygon method for faces."""
        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        # Step 2. Instantiate the RegionPolygonFile class
        polygon_file = MultiPolygon(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list
        mocked_logger = mocker.patch.object(polygon_file, "set_logger_message")

        # Step 3. Call the get_gridpoints_in_polygon method
        region_at_points = polygon_file.meshkernel_inpolygon(res_file=res_file,
                                                                 dtype="node",
                                                                 property_name="region")

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
        mocker.patch.object(polygon_file, "set_logger_message")

        # Step 3. Call the get_gridpoints_in_polygon method
        region_at_points = polygon_file.meshkernel_inpolygon(res_file=res_file, dtype="edge", property_name="region")

        # Step 4. Verify the output
        assert isinstance(region_at_points, list)
        assert len(region_at_points) == 654  # number of **internal** edges in the test file
        assert region_at_points[0] == polygon_file.undefined
        assert region_at_points[27] == "poly1"

    def test_get_points_in_polygon(self, polygon_list, mocker):
        """Test the get_points_in_polygon method."""
        # Step 1. Instantiate the RegionPolygonFile class
        polygon_file = MultiPolygon(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list
        mocker.patch.object(polygon_file, "set_logger_message")

        # Step 3. Call the get_gridpoints_in_polygon method
        region_at_points = polygon_file.get_points_in_polygon(points=[[0, 0], [200, 300]], property_name="region")

        # Step 4. Verify the output
        assert region_at_points == [polygon_file.undefined, "poly1"]

class Test_RegionPolygonFile:  # noqa: N801

    def test_initialisation(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create RegionPolygon instance
        mock_logger = mocker.patch.object(RegionPolygon, "set_logger_message")
        RegionPolygon(region_file_path=file_path, logger=logging.getLogger(__name__))

        # verify logger messages
        assert mock_logger.call_args_list[0][0][0] == "Validating region file"
        assert mock_logger.call_args_list[1][0][0] == "3 regions found"

    def test_get_gridpoints_in_polygon_creates_cache(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create RegionPolygon instance
        mocker.patch.object(RegionPolygon, "set_logger_message")
        region_polygon_file = RegionPolygon(region_file_path=file_path, logger=logging.getLogger(__name__))

        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")
        cache_file = Path(res_file).with_suffix(".region_cache.json")
        cache_file.unlink(missing_ok=True)
        # Step 2. Call the get_gridpoints_in_polygon method
        _ = region_polygon_file.get_gridpoints_in_polygon(res_file=res_file, force_cache_invalidation=True)

        # Step 3. Verify the output
        assert cache_file.exists()

    def test_get_gridpoints_in_polygon_loads_cache(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create RegionPolygon instance
        mock_logger = mocker.patch.object(RegionPolygon, "set_logger_message")
        region_polygon_file = RegionPolygon(region_file_path=file_path, logger=logging.getLogger(__name__))

        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")

        # Step 2. Call the get_gridpoints_in_polygon method
        result = region_polygon_file.get_gridpoints_in_polygon(res_file=res_file, force_cache_invalidation=False)

        # Step 3. Verify the output
        assert mock_logger.call_args_list[-1][0][0] == "Using cached regions"

        assert len(result.faces_in_polygon) == 360
        assert result.faces_in_polygon[0] == region_polygon_file.undefined
        assert result.faces_in_polygon[73] == "poly1"

        assert len(result.edges_in_polygon) == 654
        assert result.edges_in_polygon[0] == region_polygon_file.undefined
        assert result.edges_in_polygon[27] == "poly1"

    def test_get_gridpoints_in_polygon_force_cache(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create RegionPolygon instance
        mock_logger = mocker.patch.object(RegionPolygon, "set_logger_message")
        region_polygon_file = RegionPolygon(region_file_path=file_path, logger=logging.getLogger(__name__))

        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")
        cache_file = Path(res_file).with_suffix(".region_cache.json")

        # Step 2. Call the get_gridpoints_in_polygon method
        result = region_polygon_file.get_gridpoints_in_polygon(res_file=res_file, force_cache_invalidation=True)

        # Step 3. Verify the output
        assert cache_file.exists()
        mock_logger.assert_any_call("Forcing recalculating region cache", level="info")

        assert len(result.faces_in_polygon) == 360
        assert result.faces_in_polygon[0] == region_polygon_file.undefined
        assert result.faces_in_polygon[73] == "poly1"

        assert len(result.edges_in_polygon) == 654
        assert result.edges_in_polygon[0] == region_polygon_file.undefined
        assert result.edges_in_polygon[27] == "poly1"

    def test_get_gridpoints_in_polygon_stale_cache(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create RegionPolygon instance
        mock_logger = mocker.patch.object(RegionPolygon, "set_logger_message")
        region_polygon_file = RegionPolygon(region_file_path=file_path, logger=logging.getLogger(__name__))

        # Step 1. Fetch the grid file
        res_file = TestUtils.get_local_test_file("cases/case_02_compound/Data/2DModelOutput/FlowFM_map.nc")
        cache_file = Path(res_file).with_suffix(".region_cache.json")

        # update map file to change its modified time
        res_file.touch()

        # Step 2. Call the get_gridpoints_in_polygon method
        result = region_polygon_file.get_gridpoints_in_polygon(res_file=res_file, force_cache_invalidation=False)

        # Step 3. Verify the output
        assert cache_file.exists()
        mock_logger.assert_any_call("Cached regions are stale", level="warning")

        assert len(result.faces_in_polygon) == 360
        assert result.faces_in_polygon[0] == region_polygon_file.undefined
        assert result.faces_in_polygon[73] == "poly1"

        assert len(result.edges_in_polygon) == 654
        assert result.edges_in_polygon[0] == region_polygon_file.undefined
        assert result.edges_in_polygon[27] == "poly1"


class TestSectionPolygonFile:

    def test_initialisation_with_missing_section_property(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test_geojson.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create SectionPolygon instance with wrong data
        mocker.patch.object(SectionPolygon, "set_logger_message")
        with pytest.raises(PolygonError, match="Polygon poly1 has no property 'section'"):
            SectionPolygon(file_path, logger=logging.getLogger())


    def test_initialisation_with_invalid_section_name(self, mocker, test_geojson, tmp_path):
        # set up test
        file_path = tmp_path / "test_geojson.geojson"
        test_geojson["features"][0]["properties"]["section"] = "fake section"
        _geojson_file_writer(test_geojson, file_path)

        # create SectionPolygon instance with wrong data
        mocker.patch.object(SectionPolygon, "set_logger_message")
        with pytest.raises(PolygonError, match="fake section is not a recognized section"):
            SectionPolygon(file_path, logger=logging.getLogger())

    def test_initialisation_with_valid_section_polygon(self, mocker, test_geojson, tmp_path):
        # create SectionPolygon instance with correct data
        file_path = tmp_path / "test_geojson.geojson"
        mock_logger = mocker.patch.object(SectionPolygon, "set_logger_message")
        test_geojson["features"][0]["properties"]["section"] = "1"
        test_geojson["features"][1]["properties"]["section"] = "2"
        test_geojson["features"][2]["properties"]["section"] = "1"
        _geojson_file_writer(test_geojson, file_path)
        section_polygon = SectionPolygon(file_path, logging.getLogger())
        assert section_polygon.sections[0].properties["section"] == "main"
        assert section_polygon.sections[1].properties["section"] == "floodplain1"

        assert mock_logger.call_args_list[-1][0][0] == "Section file successfully validated"
