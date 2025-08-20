import json
import logging

import numpy as np
import pytest
from Pathlib import Path
from shapely.geometry import Polygon

from fm2prof.region_polygon_file import Polygon as p_tuple  # noqa: N813
from fm2prof.region_polygon_file import PolygonFile, RegionPolygonFile, SectionPolygonFile


@pytest.fixture
def polygon_list():
    return [
        p_tuple(
            geometry=Polygon([[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]),
            properties={"name": "poly1"},
        ),
        p_tuple(
            geometry=Polygon([[3, 3], [9, 3], [9, 7], [3, 7], [3, 3]]),
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
                    "coordinates": [[[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "poly2"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[3, 3], [9, 3], [9, 7], [3, 7], [3, 3]]],
                },
            },
        ],
    }


def _geojson_file_writer(geojson_dict, file_path) -> None:
    with Path(file_path).open("w") as geojson_file:
        json.dump(geojson_dict, geojson_file, indent=4)


class Test_PolygonFile:  # noqa: N801

    def test_polygonfile_parse_geojson_file(self, tmp_path, test_geojson):
        file_path = tmp_path / "polygons.geojson"

        _geojson_file_writer(test_geojson, file_path)
        polygon_file = PolygonFile(logging.getLogger())
        polygon_file.parse_geojson_file(file_path=file_path)
        assert isinstance(polygon_file.polygons[0], p_tuple)
        assert isinstance(polygon_file.polygons[0].geometry, Polygon)
        assert len(polygon_file.polygons) == 2
        assert polygon_file.polygons[1].properties.get("name") == "poly2"


    def test_polygonfile_validate_extension(self):
        polygon_file = PolygonFile(logging.getLogger())
        test_fp = "test.sjon"

        # Test invalid extension raises IOError
        with pytest.raises(IOError, match="Invalid file path extension, should be .json or .geojson."):
            polygon_file._validate_extension(file_path=test_fp)  # noqa: SLF001

        # Test valid extension does not raise
        test_fp = "test.json"
        polygon_file._validate_extension(test_fp)   # noqa: SLF001


    def test_polygonfile_check_overlap(self, polygon_list, mocker):
        polygon_file = PolygonFile(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list
        mocked_logger = mocker.patch.object(polygon_file, "set_logger_message")
        polygon_file._check_overlap()  # noqa: SLF001
        mocked_logger.assert_called_with("poly2 overlaps poly1.", level="warning")


    def test_polygonfile_polygons_property(self, polygon_list):
        polygon_file = PolygonFile(logging.getLogger())
        with pytest.raises(ValueError, match="Polygons must be of type Polygon"):
            polygon_file.polygons = ["test", "case"]

        polygon_list.append(
            p_tuple(
                geometry=Polygon([[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]),
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

class Test_RegionPolygonFile:  # noqa: N801

    def test_regionpolygonfile(self, mocker, test_geojson, tmp_path):
        # setup geojson file
        file_path = tmp_path / "test.geojson"
        _geojson_file_writer(test_geojson, file_path)

        # create RegionPolygonFile instance
        mock_logger = mocker.patch.object(RegionPolygonFile, "set_logger_message")
        region_polygon_file = RegionPolygonFile(region_file_path=file_path, logger=logging.getLogger(__name__))

        # verify logger messages
        assert mock_logger.call_args_list[0][0][0] == "Validating region file"
        assert mock_logger.call_args_list[1][0][0] == "2 regions found"

        # verify classify_points method
        xy_list = [(4, 2), (8, 6), (8, 8)]
        classified_points = region_polygon_file.classify_points(xy_list, property_name="name")
        assert np.array_equal(classified_points, ["poly1", "poly2", -999])

class Test_SectionPolygonFile:  # noqa: N801

    def test_sectionpolygonfile(self, mocker, test_geojson, tmp_path):
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

        classified_points = section_polygonfile.classify_points(points=[(4, 2), (8, 6), (8, 8)])
        assert np.array_equal(classified_points, ["main", "floodplain1", 1])
