"""Region and Section Polygon File Module.

This module provides functionality for handling polygon files used in FM2PROF
for spatial classification of 2D model data into regions and sections.

The module supports GeoJSON format polygon files and provides efficient spatial
indexing and point-in-polygon classification algorithms. It handles two main
types of polygon files:

1. **Region Polygons**: Define geographical regions for grouping cross-sections
2. **Section Polygons**: Define hydraulic sections (main channel, floodplains)

Key Features:
    - GeoJSON file parsing and validation
    - Spatial indexing using MeshKernel for efficient point classification
    - Overlap detection and validation

Classes:
    Polygon: Named tuple representing a polygon with geometry and properties.
    PolygonFile: Base class for polygon file handling with classification methods.
    RegionPolygonFile: Specialised class for region polygon files.
    SectionPolygonFile: Specialised class for section polygon files with hydraulic validation.


Note:
    Polygon files must be in GeoJSON format with specific property requirements:
    - Region polygons: require 'name' and 'id' properties
    - Section polygons: require 'name' and 'section' properties ('main', 'floodplain1', 'floodplain2')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import rtree
from netCDF4 import Dataset
from shapely.geometry import Point, shape

from fm2prof.common import FM2ProfBase

if TYPE_CHECKING:
    from collections.abc import Iterable
    from logging import Logger

    import pandas as pd


class Polygon(NamedTuple):
    """Polygon datastructure."""

    geometry: shape
    properties: dict


class PolygonFile(FM2ProfBase):
    """Polygon file class."""

    __logger = None

    def __init__(self, logger: Logger) -> None:
        """Instantiae a PolygonFile object."""
        self.set_logger(logger)
        self._polygons = []
        self.undefined = -999

    def get_gridpoints_in_polygon(self,
        res_file: str | Path,  # path to result (map) netcdf file
        dtype: Literal["face", "edge"],
        polytype: Literal["region", "section"],
    ) -> pd.DataFrame | dict:
        """Placeholder method to get points in polygon.

        1. checks if node-to-polygon has already been done
            - if so, load from file
        2. if not, perform the action. (may take several minutes)
        3. save the result to file

        TODO: node_to_face and node_to_edge should be added to input data on data import (data_import.py)

        """
        # Step 1: check if data already exists
        self.set_logger_message(f"Looking for {polytype.upper()}.dat", "debug")

        # If exist, validate and load file. If not, create it
        # to validate, store creation dates of grid and polygon file and check with input files

        # Step 2
        # get nodes_in_polygon
        # region_at_node: List[int] = meshkernel func(data)  # list of region ids for each node index [1, 1, 2, 3, ...]
        """
        # keep here
        if dtype == "face":
            node_to_face = data["face_nodes"]
            region_at_face = region_at_node[node_to_face.T[0] - 1]
            return region_at_face
        elif dtype == "edge":
            node_to_edge = data["edge_nodes"]
            region_at_edge = region_at_node[node_to_edge.T[0] - 1]
            return region_at_edge
        """

        # Step 3: save the result to file
        #self.save_to_file(polytype, region_at_face | region_at_edge) # include metadata

        # Step 4: return region_at_face or region_at_edge. This needs to be added
        # to time_independent_data or edge_data as e.g. data["region"] = region_at_face
        return #region_at_face | region_at_edge


    def classify_points_with_property(self, points: Iterable[Point], property_name: str = "name") -> np.array:
        """Classify points as belonging to which region.

        Points = list of tuples [(x,y), (x,y)]
        """
        # Convert to shapely point
        points = [Point(xy) for xy in points]
        points_regions = [self.undefined] * len(points)

        # Assign point to region
        for i, point in enumerate(points):
            for polygon in self.polygons:
                if point.within(polygon.geometry):
                    points_regions[i] = polygon.properties.get(property_name)
                    break

        return np.array(points_regions)

    def classify_points_with_property_shapely_prep(
        self,
        points: Iterable[Point],
        property_name: str = "name",
    ) -> np.array:
        """Classify points as belonging to which region.

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
                    points_regions[i] = self.polygons[p_id].properties.get(property_name)
                    break

        return np.array(points_regions)

    def classify_points_with_property_rtree_by_polygons(
        self,
        points: Iterable[Point],
        property_name: str = "name",
    ) -> list:
        """Apply RTree index to quickly classify points in polygons.

        Args:
            points (Iterable[list]): List of unformatted points.
            property_name (str): Property to retrieve from the polygons (default: {'name'})

        Returns:
            (list): List of mapped points to polygon properties.

        """
        idx = rtree.index.Index()
        for p_id, polygon in enumerate(self.polygons):
            idx.insert(p_id, polygon.geometry.bounds, polygon)

        point_properties_list = []
        for point in map(Point, points):
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

    def parse_geojson_file(self, file_path: Path | str) -> None:
        """Read data from geojson file."""
        PolygonFile._validate_extension(file_path)

        with Path(file_path).open("r") as geojson_file:
            geojson_data = json.load(geojson_file).get("features")

        for feature in geojson_data:
            feature_props = {k.lower(): v for k, v in feature.get("properties").items()}
            self.polygons.append(
                Polygon(
                    geometry=shape(feature["geometry"]).buffer(0),
                    properties=feature_props,
                ),
            )

    @staticmethod
    def _validate_extension(file_path: Path | str) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix not in (".json", ".geojson"):
            err_msg = "Invalid file path extension, should be .json or .geojson."
            raise OSError(err_msg)

    def _check_overlap(self) -> None:
        for polygon in self.polygons:
            for testpoly in self.polygons:
                if polygon.properties.get("name") == testpoly.properties.get("name"):
                    # polygon will obviously overlap with itself
                    continue
                if polygon.geometry.intersects(testpoly.geometry):
                    self.set_logger_message(
                        "{} overlaps {}.".format(
                            polygon.properties.get("name"),
                            testpoly.properties.get("name"),
                        ),
                        level="warning",
                    )

    @property
    def polygons(self) -> list[Polygon]:
        """Polygons."""
        return self._polygons

    @polygons.setter
    def polygons(self, polygons_list: list[Polygon]) -> None:
        if not all([isinstance(polygon, Polygon) for polygon in polygons_list]):  # noqa: C419
            err_msg = "Polygons must be of type Polygon"
            raise ValueError(err_msg)
        # Check if properties contain the required 'name' property
        names = [polygon.properties.get("name") for polygon in polygons_list]
        if not all(names):
            err_msg = "Polygon properties must contain key-word 'name'"
            raise ValueError(err_msg)
        # Check if 'name' property is unique, otherwise _check_overlap will produce bugs
        if len(names) != len(set(names)):
            err_msg = "Property 'name' must be unique"
            raise ValueError(err_msg)
        self._polygons = polygons_list


class RegionPolygonFile(PolygonFile):
    """RegionPolygonFile class."""

    def __init__(self, region_file_path: str | Path, logger: Logger) -> None:
        """Instantiate a RegionPolygonFile object."""
        super().__init__(logger)
        self.read_region_file(region_file_path)

    @property
    def regions(self) -> list[Polygon]:
        """Region polygons."""
        return self.polygons

    def read_region_file(self, file_path: Path | str) -> None:
        """Read region file.

        Args:
            file_path (Path | str): region file path

        """
        self.parse_geojson_file(file_path)
        self._validate_regions()

    def _validate_regions(self) -> None:
        self.set_logger_message("Validating region file", level="info")

        number_of_regions = len(self.regions)

        self.set_logger_message(f"{number_of_regions} regions found", level="info")

        # Test if polygons overlap
        self._check_overlap()

    def classify_points(self, points: Iterable[Point], property_name: str = "id") -> list:
        """Classify region points with a property.

        Args:
            points (Iterable[Point]): Points to classify
            property_name (str, optional): Property. Defaults to "id".

        Returns:
            list: _description_

        """
        return self.classify_points_with_property(points, property_name=property_name)


class SectionPolygonFile(PolygonFile):
    """SectionPolygonFile class."""

    def __init__(self, section_file_path: str | Path, logger: Logger) -> None:
        """Instantiate a SectionPolygonFile object.

        Args:
            section_file_path (str | Path): path to section polygon file.
            logger (Logger): logger

        """
        super().__init__(logger)
        self.read_section_file(section_file_path)
        self.undefined = 1  # 1 is main

    @property
    def sections(self) -> list[Polygon]:
        """Section polygons."""
        return self.polygons

    def read_section_file(self, file_path: str | Path) -> None:
        """Read section polygon file.

        Args:
            file_path (str | Path): path to section polygon file.

        """
        self.parse_geojson_file(file_path)
        self._validate_sections()

    def classify_points(self, points: Iterable[Point]) -> np.array:
        """Classify points with a section property name.

        Args:
            points (Iterable[Point]): List of points to classify.

        Returns:
            np.array: array of points

        """
        return self.classify_points_with_property(points, property_name="section")

    def _validate_sections(self) -> None:
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
                    'Polygon {} has no property "section"'.format(section.properties.get("name")),
                    level="error",
                )

            elif str(section.properties.get("section")).lower() not in valid_section_keys:
                section_key = str(section.properties.get("section")).lower()
                if section_key not in list(map_section_keys.keys()):
                    raise_exception = True
                    self.set_logger_message(
                        f"{section_key} is not a recognized section",
                        level="error",
                    )
                else:
                    self.set_logger_message(
                        f"remapped section {section_key} to {map_section_keys[section_key]}",
                        level="warning",
                    )
                    section.properties["section"] = map_section_keys.get(section_key)
        # check for overlap (only raise a warning)
        self._check_overlap()

        if raise_exception:
            err_msg = "Section file is not valid"
            raise OSError(err_msg)
        self.set_logger_message("Section file succesfully validated")
