import json
import logging
import os
import timeit
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import fixture
from shapely.geometry import Polygon

from fm2prof.region_polygon_file import Polygon as p_tuple
from fm2prof.region_polygon_file import PolygonFile, RegionPolygonFile, SectionPolygonFile
from tests.TestUtils import TestUtils


class ARCHIVED_Test_PolygonFile:
    class ClassifierApproaches:
        def __init__(self, polygon_file, xy_list):
            self.polygon_file = polygon_file
            self.xy_list = xy_list

        def test_action_regular(self):
            return self.polygon_file.classify_points_with_property(self.xy_list)

        def test_action_regular_prep(self):
            return self.polygon_file.classify_points_with_property_shapely_prep(iter(self.xy_list))

        def test_action_polygons(self):
            return self.polygon_file.classify_points_with_property_rtree_by_polygons(self.xy_list)

        @staticmethod
        def test_action_regular_static(polygon_file, xy_list):
            return polygon_file.classify_points_with_property(xy_list)

        @staticmethod
        def test_action_prep_static(polygon_file, xy_list):
            return polygon_file.classify_points_with_property_shapely_prep(iter(xy_list))

        @staticmethod
        def test_action_polygons_static(polygon_file, xy_list):
            return polygon_file.classify_points_with_property_rtree_by_polygons(xy_list)

    def __get_basic_polygon_list(self, left_classifier, right_classifier):
        geometry_left = p_tuple(
            geometry=Polygon([[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]),
            properties={"name": left_classifier},
        )
        geometry_right = p_tuple(
            geometry=Polygon([[3, 3], [9, 3], [9, 7], [3, 7], [3, 3]]),
            properties={"name": right_classifier},
        )
        return [geometry_left, geometry_right]

    def __get_basic_point_list(self):
        left_point = (3, 2)
        right_point = (5, 5)
        overlap_point = (4, 4)
        outside_point_left = (2, 5)
        outside_point_right = (6, 2)
        return [
            left_point,
            right_point,
            overlap_point,
            outside_point_left,
            outside_point_right,
        ]

    def __get_random_point(self, max_x, max_y):
        x_pos = randint(0, max_x)
        y_pos = randint(0, max_y)
        return (x_pos, y_pos)

    @pytest.mark.acceptance
    @pytest.mark.parametrize(
        "classifier_function",
        [
            (ClassifierApproaches.test_action_regular_static),
            (ClassifierApproaches.test_action_prep_static),
            (ClassifierApproaches.test_action_polygons_static),
        ],
    )
    def test_given_list_of_geometries_then_classifies_correctly(self, classifier_function):
        # 1. Defining test input data
        left_classifier = "left_classifier"
        right_classifier = "right_classifier"
        undefined_classifier = "undefined"
        # 1.1. Prepare polygons
        geometry_left = p_tuple(
            geometry=Polygon([[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]),
            properties={"name": left_classifier},
        )
        geometry_right = p_tuple(
            geometry=Polygon([[3, 3], [9, 3], [9, 7], [3, 7], [3, 3]]),
            properties={"name": right_classifier},
        )
        polygon_list = [geometry_left, geometry_right]
        # 1.2. Prepare points
        left_point = (3, 2)
        right_point = (5, 5)
        overlap_point = (4, 4)
        outside_point_left = (2, 5)
        outside_point_right = (6, 2)
        xy_list = [
            left_point,
            right_point,
            overlap_point,
            outside_point_left,
            outside_point_right,
        ]
        expected_classifiers = [
            left_classifier,
            right_classifier,
            left_classifier,
            undefined_classifier,
            undefined_classifier,
        ]
        np_expected_classifiers = np.array(expected_classifiers)
        # 2. Verify initial expectations.
        polygon_file = PolygonFile(logging.getLogger(__name__))
        assert polygon_file is not None
        polygon_file.polygons = polygon_list

        # 3. Run test
        classifiers = classifier_function(polygon_file, xy_list)

        # 4. Verify final expectations.
        assert classifiers is not None
        assert np.array_equal(classifiers, np_expected_classifiers)

    @pytest.mark.acceptance
    @pytest.mark.parametrize("number_of_points", [(10), (100), (1000), (10000)])
    def test_overall_performance(self, number_of_points: int):
        # 1. Defining test input data
        left_classifier = "left"
        right_classifier = "right"
        undefined_classifier = "undefined"
        classifiers_names = [left_classifier, right_classifier, undefined_classifier]
        polygon_list = self.__get_basic_polygon_list(left_classifier, right_classifier)
        map_boundary = (10, 10)
        xy_list = [self.__get_random_point(map_boundary[0], map_boundary[1]) for _ in range(number_of_points)]
        polygon_file = PolygonFile(logging.getLogger(__name__))
        polygon_file.polygons = polygon_list

        # 3. Run test
        self.__run_performance_test(polygon_file, xy_list, classifiers_names)

    @pytest.mark.acceptance
    def test_classify_points_for_waal(self):
        # 1. Set up test data.
        # Get polygons data.
        section_data_dir = TestUtils.get_local_test_data_dir("performance_waal")
        section_file = os.path.join(section_data_dir, "SectionPolygonDissolved.json")
        assert os.path.exists(section_file)

        self.__logger = logging.getLogger(__name__)
        polygon = SectionPolygonFile(section_file, logger=self.__logger)
        assert polygon is not None

        # Read NC File
        waal_data_dir = TestUtils.get_external_test_data_subdir("case_08_waal") / "Data" / "FM"
        waal_nc_file = waal_data_dir / "FlowFM_fm2prof_map.nc"
        assert waal_nc_file.is_file()

        _, edge_data, _, _ = TestUtils.read_fm_model(str(waal_nc_file))
        points = [(edge_data["x"][i], edge_data["y"][i]) for i in range(len(edge_data["x"]))]
        assert points is not None

        # 2. Run test
        self.__run_performance_test(polygon, points, None)

    def __run_performance_test(self, polygon_file: PolygonFile, xy_list: list, classifiers_names: list):
        number_of_points = len(xy_list)
        t_repetitions = 10

        def time_function(function_name) -> list:
            # buffer
            return timeit.repeat(function_name, repeat=t_repetitions, number=1000)

        ca = self.ClassifierApproaches(polygon_file, xy_list)
        cases_list = {
            "regular": ca.test_action_regular,
            "shapely-prep": ca.test_action_regular_prep,
            # 'r-tree': ca.test_action_polygons,
        }

        import itertools

        t_results = {}
        c_results = {}
        for case_name, case_func in cases_list.items():
            t_results[case_name] = time_function(case_func)
            values = case_func()
            c_results[case_name] = values

        if not classifiers_names:
            return
        # Plot reults
        output_dir = TestUtils.get_test_dir_output("PolygonFile_Performance")
        markers = itertools.cycle((",", "+", ".", "o", "*"))
        plt.figure()
        for name, result in t_results.items():
            plt.plot(range(t_repetitions), result, label=name)
        plt.legend()
        plt.savefig(output_dir + f"\\time_performance_points_{number_of_points}.png")
        plt.close()

        plt.figure()
        for name, result in c_results.items():
            values = [classifiers_names.index(val) for val in list(result)]
            plt.scatter(range(len(list(result))), values, marker=next(markers), label=name)
        plt.yticks(
            list(range(len(classifiers_names))),
            classifiers_names,
            rotation=45,
        )
        plt.legend()
        plt.savefig(output_dir + f"\\classifier_results_points_{number_of_points}.png")
        plt.close()


@fixture
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


@fixture
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


def _geojson_file_writer(geojson_dict, file_path):
    with open(file_path, "w") as geojson_file:
        json.dump(geojson_dict, geojson_file, indent=4)


def test_PolygonFile_classify_points_with_property(polygon_list):
    polygon_file = PolygonFile(logging.getLogger())
    polygon_file.polygons = polygon_list
    xy_list = [(4, 2), (8, 6), (8, 8)]

    classified_points = polygon_file.classify_points_with_property(points=xy_list)
    assert np.array_equal(classified_points, ["poly1", "poly2", -999])


def test_PolygonFile_classify_points_with_property_shapely_prep(polygon_list):
    polygon_file = PolygonFile(logging.getLogger())
    polygon_file.polygons = polygon_list
    xy_list = [(4, 2), (8, 6), (8, 8)]
    classified_points = polygon_file.classify_points_with_property_shapely_prep(points=xy_list, property_name="name")
    assert np.array_equal(classified_points, ["poly1", "poly2", -999])


def test_PolygonFile_classify_points_with_property_rtree_by_polygons(polygon_list):
    polygon_file = PolygonFile(logging.getLogger())
    polygon_file.polygons = polygon_list
    xy_list = [(4, 2), (8, 6), (8, 8)]
    classified_points = polygon_file.classify_points_with_property_rtree_by_polygons(
        points=xy_list,
        property_name="name",
    )
    assert np.array_equal(classified_points, ["poly1", "poly2", -999])


def test_PolygonFile_parse_geojson_file(tmp_path, test_geojson):
    file_path = tmp_path / "polygons.geojson"

    _geojson_file_writer(test_geojson, file_path)
    polygon_file = PolygonFile(logging.getLogger())
    polygon_file.parse_geojson_file(file_path=file_path)
    assert isinstance(polygon_file.polygons[0], p_tuple)
    assert isinstance(polygon_file.polygons[0].geometry, Polygon)
    assert len(polygon_file.polygons) == 2
    assert polygon_file.polygons[1].properties.get("name") == "poly2"


def test_PolygonFile_validate_extension():
    polygon_file = PolygonFile(logging.getLogger())
    test_fp = "test.sjon"

    with pytest.raises(IOError, match="Invalid file path extension, should be .json or .geojson."):
        polygon_file._validate_extension(file_path=test_fp)
    test_fp = "test.json"
    polygon_file._validate_extension(test_fp)


def test_PolygonFile_check_overlap(polygon_list, mocker):
    polygon_file = PolygonFile(logging.getLogger(__name__))
    polygon_file.polygons = polygon_list
    mocked_logger = mocker.patch.object(polygon_file, "set_logger_message")
    polygon_file._check_overlap()
    mocked_logger.assert_called_with("poly2 overlaps poly1.", level="warning")


def test_PolygonFile_polygons_property(polygon_list):
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


def test_RegionPolygonFile(mocker, test_geojson, tmp_path):
    file_path = tmp_path / "test.geojson"
    _geojson_file_writer(test_geojson, file_path)
    mock_logger = mocker.patch.object(RegionPolygonFile, "set_logger_message")
    region_polygon_file = RegionPolygonFile(region_file_path=file_path, logger=logging.getLogger(__name__))
    assert mock_logger.call_args_list[0][0][0] == "Validating region file"
    assert mock_logger.call_args_list[1][0][0] == "2 regions found"

    xy_list = [(4, 2), (8, 6), (8, 8)]
    classified_points = region_polygon_file.classify_points(xy_list, property_name="name")
    assert np.array_equal(classified_points, ["poly1", "poly2", -999])


def test_SectionPolygonFile(mocker, test_geojson, tmp_path):
    file_path = tmp_path / "test_geojson.geojson"
    _geojson_file_writer(test_geojson, file_path)
    mock_logger = mocker.patch.object(SectionPolygonFile, "set_logger_message")
    with pytest.raises(OSError, match="Section file is not valid"):
        SectionPolygonFile(file_path, logger=logging.getLogger())

        assert mock_logger.call_args_list[1][0][0] == 'Polygon poly1 has no property "section"'
        assert mock_logger.call_args_list[2][0][0] == 'Polygon poly2 has no property "section"'

    test_geojson["features"][0]["properties"]["section"] = "fake section"
    _geojson_file_writer(test_geojson, file_path)
    with pytest.raises(OSError, match="Section file is not valid"):
        SectionPolygonFile(file_path, logger=logging.getLogger())
        assert "fake section is not a recognized section" in [log_cal[0][0] for log_cal in mock_logger.call_args_list]
    test_geojson["features"][0]["properties"]["section"] = "1"
    test_geojson["features"][1]["properties"]["section"] = "2"
    _geojson_file_writer(test_geojson, file_path)
    section_polygonfile = SectionPolygonFile(file_path, logging.getLogger())
    assert section_polygonfile.sections[0].properties["section"] == "main"
    assert section_polygonfile.sections[1].properties["section"] == "floodplain1"

    assert mock_logger.call_args_list[-1][0][0] == "Section file succesfully validated"

    classified_points = section_polygonfile.classify_points(points=[(4, 2), (8, 6), (8, 8)])
    assert np.array_equal(classified_points, ["main", "floodplain1", 1])
