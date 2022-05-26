"""
Copyright (C) Stichting Deltares 2019. All rights reserved.

This file is part of the Fm2Prof.

The Fm2Prof is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.
"""

import json
import logging
import multiprocessing
import os
from collections import namedtuple
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Point, shape

from fm2prof.common import FM2ProfBase

Polygon = namedtuple("Polygon", ["geometry", "properties"])


class PolygonFile(FM2ProfBase):
    __logger = None

    def __init__(self, logger):
        self.set_logger(logger)
        self.polygons = list()
        self.undefined = -999

    def classify_points_with_property(
        self, points: Iterable[list], property_name: str = "name"
    ) -> np.array:
        """
        Classifies points as belonging to which region

        Points = list of tuples [(x,y), (x,y)]
        """
        # Convert to shapely point
        points = [Point(xy) for xy in points]
        points_regions = [self.undefined] * len(points)

        # Assign point to region
        for i, point in enumerate(points):
            for polygon in self.polygons:
                if point.within(polygon.geometry):
                    points_regions[i] = int(polygon.properties.get(property_name))
                    break

        return np.array(points_regions)

    def classify_points_with_property_shapely_prep(
        self, points: Iterable[list], property_name: str = "name"
    ):
        """
        Classifies points as belonging to which region

        Points = list of tuples [(x,y), (x,y)]
        """
        from shapely.prepared import prep

        # Convert to shapely point
        points = [Point(xy) for xy in points]
        points_regions = [self.undefined] * len(points)

        prep_polygons = [prep(p.geometry) for p in self.polygons]

        # Assign point to region
        for i, point in enumerate(points):
            for p_id, polygon in enumerate(prep_polygons):
                if polygon.intersects(point):
                    points_regions[i] = self.polygons[p_id].properties.get(
                        property_name
                    )
                    break

        return np.array(points_regions)

    def classify_points_with_property_rtree_by_polygons(
        self, iterable_points: Iterable[list], property_name: str = "name"
    ) -> list:
        """Applies RTree index to quickly classify points in polygons.

        Arguments:
            iterable_points {Iterable[list]} -- List of unformatted points.

        Keyword Arguments:
            property_name {str}
                -- Property to retrieve from the polygons (default: {'name'})

        Returns:
            list -- List of mapped points to polygon properties.
        """
        idx = rtree.index.Index()
        for p_id, polygon in enumerate(self.polygons):
            idx.insert(p_id, polygon.geometry.bounds, polygon)

        point_properties_list = []
        for point in map(Point, iterable_points):
            point_properties_polygon = next(
                iter(
                    self.polygons[polygon_id].properties.get(property_name)
                    for polygon_id in idx.intersection(point.bounds)
                    if self.polygons[polygon_id].geometry.intersects(point)
                ),
                self.undefined,
            )
            point_properties_list.append(point_properties_polygon)

        del idx
        return np.array(point_properties_list)

    def __get_polygon_property(self, grouped_values: list, property_name: str) -> str:
        """Retrieves the polygon property from the internal list of polygons.

        Arguments:
            grouped_values {int}
                -- Grouped values containing point and polygon id.
            property_name {str} -- Property to search.

        Returns:
            str -- Property value.
        """
        polygon_id = list(grouped_values[1])[0][0]
        if polygon_id >= len(self.polygons) or polygon_id < 0:
            return self.undefined
        return self.polygons[polygon_id].properties.get(property_name)

    def parse_geojson_file(self, file_path):
        """Read data from geojson file"""
        PolygonFile._validate_extension(file_path)

        with open(file_path) as geojson_file:
            geojson_data = json.load(geojson_file).get("features")

        for feature in geojson_data:
            feature_props = {k.lower(): v for k, v in feature.get("properties").items()}
            self.polygons.append(
                Polygon(
                    geometry=shape(feature["geometry"]).buffer(0),
                    properties=feature_props,
                )
            )
        # polygons = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in geojson_data])
        # polygon_names = [feature.get('properties').get('Name') for feature in geojson_data]

        # for polygon, polygon_name in zip(polygons, polygon_names):
        #    self.polygons[polygon_name] = polygon

    @staticmethod
    def _validate_extension(file_path: Path) -> None:
        if not isinstance(file_path, Path):
            return
        if not file_path.suffix in (".json", ".geojson"):
            raise IOError(
                "Invalid file path extension, " + "should be .json or .geojson."
            )

    def _check_overlap(self):
        for polygon in self.polygons:
            for testpoly in self.polygons:
                if polygon.properties.get("name") == testpoly.properties.get("name"):
                    # polygon will obviously overlap with itself
                    pass
                else:
                    if polygon.geometry.intersects(testpoly.geometry):
                        self.set_logger_message(
                            "{} overlaps {}.".format(
                                polygon.properties.get("name"),
                                testpoly.properties.get("name"),
                            ),
                            level="warning",
                        )


class RegionPolygonFile(PolygonFile):
    def __init__(self, region_file_path, logger):
        super().__init__(logger)
        self.read_region_file(region_file_path)

    @property
    def regions(self):
        return self.polygons

    def read_region_file(self, file_path):
        self.parse_geojson_file(file_path)
        self._validate_regions()

    def _validate_regions(self):
        self.set_logger_message("Validating Region file")

        number_of_regions = len(self.regions)

        self.set_logger_message("{} regions found".format(number_of_regions))

        # Test if polygons overlap
        self._check_overlap()

    def classify_points(self, points: Iterable[list]):
        return self.classify_points_with_property(points, property_name="id")


class SectionPolygonFile(PolygonFile):
    def __init__(self, section_file_path, logger):
        super().__init__(logger)
        self.read_section_file(section_file_path)
        self.undefined = 1  # 1 is main

    @property
    def sections(self):
        return self.polygons

    def read_section_file(self, file_path):
        self.parse_geojson_file(file_path)
        self._validate_sections()

    def classify_points(self, points: Iterable[list]):
        return self.classify_points_with_property(points, property_name="section")

    def _validate_sections(self):
        self.set_logger_message("Validating Section file")
        raise_exception = False

        valid_section_keys = {"main", "floodplain1", "floodplain2"}
        map_section_keys = {
            "1": "main",
            "2": "floodplain1",
            "3": "floodplain2",
        }

        # each polygon must have a 'section' property
        for section in self.sections:
            if "section" not in section.properties:
                raise_exception = True
                self.set_logger_message(
                    'Polygon {} has no property "section"'.format(
                        section.properties.get("name")
                    ),
                    level="error",
                )
            section_key = str(section.properties.get("section")).lower()
            if section_key not in valid_section_keys:
                if section_key not in list(map_section_keys.keys()):
                    raise_exception = True
                    self.set_logger_message(
                        "{} is not a recognized section".format(section_key),
                        level="error",
                    )
                else:
                    self.set_logger_message(
                        "remapped section {} to {}".format(
                            section_key, map_section_keys[section_key]
                        ),
                        level="warning",
                    )
                    section.properties["section"] = map_section_keys.get(section_key)
        # check for overlap (only raise a warning)
        self._check_overlap()

        if raise_exception:
            raise AssertionError("Section file could not validated")
        else:
            self.set_logger_message("Section file succesfully validated")
