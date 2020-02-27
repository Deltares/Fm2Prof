import pytest
import sys
import os
import logging
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from random import seed, randint
import timeit
import time

from fm2prof.RegionPolygonFile import PolygonFile
from fm2prof.RegionPolygonFile import Polygon as p_tuple
from tests.TestUtils import TestUtils


class Test_PolygonFile:

    @pytest.mark.acceptance
    def test_given_list_of_geometries_then_classifies_correctly(self):
        # 1. Defining test input data
        left_classifier = 'left_classifier'
        right_classifier = 'right_classifier'
        undefined_classifier = 'undefined'
        # 1.1. Prepare polygons
        geometry_left = \
            p_tuple(
                geometry=Polygon([[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]),
                properties={'name': left_classifier})
        geometry_right = \
            p_tuple(
                geometry=Polygon([[3, 3], [9, 3], [9, 7], [3, 7], [3, 3]]),
                properties={'name': right_classifier})
        polygon_list = [geometry_left, geometry_right]
        # 1.2. Prepare points
        left_point = (3, 2)
        right_point = (5, 5)
        overlap_point = (4, 4)
        outside_point_left = (2, 5)
        outside_point_right = (6, 2)
        xy_list = [
            left_point, right_point,
            overlap_point,
            outside_point_left, outside_point_right]
        expected_classifiers = [
            left_classifier, right_classifier,
            left_classifier,
            undefined_classifier, undefined_classifier]
        # 2. Verify initial expectations.
        polygon_file = PolygonFile(
            logging.getLogger(__name__))
        assert polygon_file is not None
        polygon_file.polygons = polygon_list

        # 3. Run test
        classifiers = \
            polygon_file.classify_points_with_property_rtree_by_polygons(
                xy_list)

        # 4. Verify final expectations.
        assert classifiers is not None
        assert classifiers == expected_classifiers

    class ClassifierApproaches:

        def __init__(self, polygon_file, xy_list):
            self.polygon_file = polygon_file
            self.xy_list = xy_list

        def test_action_regular(self):
            return \
                self.polygon_file.classify_points_with_property(
                    self.xy_list)

        def test_action_regular_optimized(self):
            return \
                self.polygon_file.classify_points_with_property_optimized(
                    iter(self.xy_list))

        def test_action_regular_prep(self):
            return \
                self.polygon_file.classify_points_with_property_shapely_prep(
                    iter(self.xy_list))

        def test_action_polygons(self):
            return \
                self.polygon_file.classify_points_with_property_rtree_by_polygons(
                    self.xy_list)

    def __get_basic_polygon_list(self, left_classifier, right_classifier):
        geometry_left = \
            p_tuple(
                geometry=Polygon([[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]]),
                properties={'name': left_classifier})
        geometry_right = \
            p_tuple(
                geometry=Polygon([[3, 3], [9, 3], [9, 7], [3, 7], [3, 3]]),
                properties={'name': right_classifier})
        return [geometry_left, geometry_right]

    def __get_basic_point_list(self):
        left_point = (3, 2)
        right_point = (5, 5)
        overlap_point = (4, 4)
        outside_point_left = (2, 5)
        outside_point_right = (6, 2)
        return [
            left_point, right_point,
            overlap_point,
            outside_point_left, outside_point_right]

    def __get_random_point(self, max_x, max_y):
        x_pos = randint(0, max_x)
        y_pos = randint(0, max_y)
        return (x_pos, y_pos)

    @pytest.mark.acceptance
    @pytest.mark.parametrize(
        'number_of_points',
        [(10), (100), (1000), (10000)])
    def test_performance(self, number_of_points: int):
        # 1. Defining test input data
        left_classifier = 'left'
        right_classifier = 'right'
        undefined_classifier = 'undefined'
        classifiers_names = \
            [left_classifier, right_classifier, undefined_classifier]
        polygon_list = \
            self.__get_basic_polygon_list(
                left_classifier,
                right_classifier)
        map_boundary = (10, 10)
        xy_list = [
            self.__get_random_point(map_boundary[0], map_boundary[1])
            for _ in range(number_of_points)]
        polygon_file = PolygonFile(
            logging.getLogger(__name__))
        polygon_file.polygons = polygon_list

        # 3. Run test
        t_repetitions = 10

        def time_function(function_name) -> list:
            # buffer
            return timeit.repeat(
                function_name,
                repeat=t_repetitions,
                number=1000)

        ca = self.ClassifierApproaches(polygon_file, xy_list)
        cases_list = {
            'regular': ca.test_action_regular,
            'optimized': ca.test_action_regular_optimized,
            'shapely-prep': ca.test_action_regular_prep,
            'r-tree polygons': ca.test_action_polygons,
        }
        import itertools
        t_results = {}
        c_results = {}
        for case_name, case_func in cases_list.items():
            t_results[case_name] = time_function(case_func)
            values = case_func()
            c_results[case_name] = values

        # Plot reults
        output_dir = TestUtils.get_test_dir_output('PolygonFile_Performance')
        markers = itertools.cycle((',', '+', '.', 'o', '*'))
        plt.figure()
        for name, result in t_results.items():
            plt.plot(
                range(t_repetitions),
                result,
                label=name)
        plt.legend()
        plt.savefig(
            output_dir
            + '\\time_performance_points_{}.png'.format(number_of_points))
        plt.close()

        plt.figure()
        for name, result in c_results.items():
            values = [classifiers_names.index(val) for val in list(result)]
            plt.scatter(
                range(len(list(result))),
                values,
                marker=next(markers),
                label=name)
        plt.yticks(
            [c_id for c_id in range(len(classifiers_names))],
            classifiers_names,
            rotation=45)
        plt.legend()
        plt.savefig(
            output_dir
            + '\\classifier_results_points_{}.png'.format(number_of_points))
        plt.close()
