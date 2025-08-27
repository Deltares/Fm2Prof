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
from typing import TYPE_CHECKING, Literal

import shapely
from meshkernel import GeometryList, MeshKernel, ProjectionType

from fm2prof.common import FM2ProfBase
from fm2prof.data_import import FMDataImporter

if TYPE_CHECKING:
    from logging import Logger

    import numpy as np
    import pandas as pd

class Polygon:
    """Polygon class."""

    def __init__(self, coordinates: list[list[float]], properties: dict) -> None:
        """Instantiate a Polygon object.

        Args:
            coordinates (list[list[float]]): List of [x, y] coordinates defining the polygon.
            properties (dict): Dictionary of properties associated with the polygon.
        """
        self.coordinates = coordinates
        self.properties = properties

    @property
    def x(self) -> list[float]:
        """X coordinates of the polygon."""
        return [coord[0] for coord in self.coordinates]

    @property
    def y(self) -> list[float]:
        """Y coordinates of the polygon."""
        return [coord[1] for coord in self.coordinates]


class MultiPolygon(FM2ProfBase):
    """MultiPolygon file class.

    This class handles MultiPolygon files used in FM2PROF for spatial classification
    of 2D model data into regions and sections. It supports GeoJSON format polygon files
    and provides methods for reading, validating, and classifying points within the polygons.

    We use a base MultiPolygon class to leverage MeshKernel and Shapely functionality
    within a common framework. E.g. `MultiPolygon.as_meshkernel()` outputs a MeshKernel GeometryList
    object that can be used for spatial classification, while `MultiPolygon.as_shapely()` outputs
    a list of Shapely Polygon objects for geometric operations.
    """

    __logger = None

    def __init__(self, logger: Logger) -> None:
        """Instantiate a MultiPolygon object."""
        self.set_logger(logger)
        self._polygons = []
        self.undefined = "undefined"

    @property
    def polygons(self) -> list[Polygon]:
        """Polygons."""
        return self._polygons

    @polygons.setter
    def polygons(self, polygons_list: list[Polygon]) -> None:
        if not all(isinstance(polygon, Polygon) for polygon in polygons_list):
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

    def as_meshkernel(self) -> list[GeometryList]:
        """Convert polygons to MeshKernel GeometryList.

        Note:
            MeshKernel GeometryList supports multiple polygons,
            separated by some int (default -999). However,
            to keep track of polygon properties (e.g. name),
            we create a list of single polygon GeometryList objects.

        Returns:
            GeometryList: MeshKernel GeometryList object containing all polygons.
        """
        if not self.polygons:
            err_msg = "No polygons defined"
            raise ValueError(err_msg)

        return [GeometryList(x_coordinates=polygon.x, y_coordinates=polygon.y) for polygon in self.polygons]

    def as_shapely(self) -> list[shapely.geometry.Polygon]:
        """Convert polygons to list of Shapely Polygon objects.

        Returns:
            list[shapely.geometry.Polygon]: List of Shapely Polygon objects.
        """
        if not self.polygons:
            err_msg = "No polygons defined"
            raise ValueError(err_msg)

        return [shapely.Polygon(polygon.coordinates) for polygon in self.polygons]

    def get_gridpoints_in_polygon(self,
        res_file: str | Path,  # path to result (map) netcdf file
        dtype: Literal["face", "edge", "node"],
        property_name: Literal["name", "section"],
        ) -> list[str]:
        """Get grid points in polygon.

        Args:
            res_file (str | Path): Path to result (map) netcdf file.
            dtype (Literal["face", "edge", "node"]): Type of grid points to retrieve.
            property_name (Literal["name", "section"]): Property to use for classification.

        Returns:
            pd.DataFrame | dict: DataFrame or dictionary containing grid points in polygon.
        """
        # Step 1
        # Construct Meshkernel grid
        mk = MeshKernel(projection=ProjectionType.CARTESIAN)
        mesh2d_input = mk.mesh2d_get()

        fmdata = FMDataImporter(res_file)
        mesh2d_input.node_x = fmdata.get_variable("mesh2d_node_x")
        mesh2d_input.node_y = fmdata.get_variable("mesh2d_node_y")
        mesh2d_input.edge_nodes = fmdata.get_variable("mesh2d_edge_nodes").flatten() - 1

        mk.mesh2d_set(mesh2d_input)

        # Step 2: perform point-in-polygon classification
        nodes_in_polygon: list[str] = [self.undefined] * len(mesh2d_input.node_x)  # default to undefined

        for i, mk_polygon in enumerate(self.as_meshkernel()):
            indices: list[float] = mk.mesh2d_get_nodes_in_polygons(mk_polygon, inside=True).tolist()
            for j in indices:
                nodes_in_polygon[j] = self.polygons[i].properties.get(property_name)

        # Step 3: map to faces or edges if needed
        if dtype == "node":
            output = nodes_in_polygon
        elif dtype == "face":
            face_map: np.ndarray = fmdata.get_variable("mesh2d_face_nodes").T[0] -1
            node_to_face: list[str] = [self.undefined] * len(face_map)
            for face_index, map_index in enumerate(face_map.tolist()):
                node_to_face[int(face_index)] = nodes_in_polygon[int(map_index)]
            output = node_to_face
        elif dtype == "edge":
            edge_map: np.ndarray = fmdata.get_variable("mesh2d_edge_nodes").T[0] -1
            node_to_edge: list[str] = [self.undefined] * len(edge_map)
            for edge_index, map_index in enumerate(edge_map.tolist()):
                node_to_edge[int(edge_index)] = nodes_in_polygon[int(map_index)]
            output = node_to_edge

        return output

    def check_overlap(self) -> None:
        """Check if polygons overlap and log a warning if they do."""
        for i, poly1 in enumerate(self.as_shapely()):
            for j, poly2 in enumerate(self.as_shapely()):
                if i == j:
                    # polygon will obviously overlap with itself
                    continue
                if poly1.intersects(poly2):
                    self.set_logger_message(
                        f"{self.polygons[i].properties.get('name')} overlaps {self.polygons[j].properties.get('name')}." ,
                        level="warning",
                    )

    def from_file(self, file_path: Path | str) -> None:
        """Read data from geojson file.

        Args:
            file_path (Path | str): path to geojson file
        """
        self._validate_extension(file_path)

        with Path(file_path).open("r") as geojson_file:
            geojson_data = json.load(geojson_file).get("features")

        polygons: list[Polygon] = []
        for feature in geojson_data:
            feature_props = {k.lower(): v for k, v in feature.get("properties").items()}
            polygons.append(Polygon(coordinates=feature["geometry"]["coordinates"], properties=feature_props))

        self.polygons = polygons

    @staticmethod
    def _validate_extension(file_path: Path | str) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix not in (".json", ".geojson"):
            err_msg = "Invalid file path extension, should be .json or .geojson."
            raise OSError(err_msg)


class RegionPolygonFile(MultiPolygon):
    """RegionPolygonFile class."""

    def __init__(self, region_file_path: str | Path, logger: Logger) -> None:
        """Instantiate a RegionPolygonFile object."""
        super().__init__(logger)
        self.from_file(region_file_path)

    @property
    def regions(self) -> list[Polygon]:
        """Region polygons."""
        return self.polygons

    def from_file(self, file_path: Path | str) -> None:
        """Read geojson file and performs validation.

        Args:
            file_path (Path | str): region file path

        """
        super().from_file(file_path)
        self._validate_regions()

    def get_gridpoints_in_polygon(self, res_file: str | Path) -> list[str]:
        """Convenience method to get faces in region.

        This method is an overload of the parent method with fixed
        property_name and dtype arguments.

        Args:
            res_file (str | Path): path to result (map) netcdf file.

        Returns:
            list[str]: List of region names for each face in the grid.
        """
        super().get_gridpoints_in_polygon(res_file, "face", "name")

    def _validate_regions(self) -> None:
        self.set_logger_message("Validating region file", level="info")

        number_of_regions = len(self.regions)

        self.set_logger_message(f"{number_of_regions} regions found", level="info")

        # Test if polygons overlap
        self.check_overlap()


class SectionPolygonFile(MultiPolygon):
    """SectionPolygonFile class."""

    def __init__(self, section_file_path: str | Path, logger: Logger) -> None:
        """Instantiate a SectionPolygonFile object.

        Args:
            section_file_path (str | Path): path to section polygon file.
            logger (Logger): logger

        """
        super().__init__(logger)
        self.from_file(section_file_path)
        self.undefined = 1  # 1 is main

    @property
    def sections(self) -> list[Polygon]:
        """Section polygons."""
        return self.polygons

    def from_file(self, file_path: str | Path) -> None:
        """Read section polygon file.

        Args:
            file_path (str | Path): path to section polygon file.

        """
        super().from_file(file_path)
        self._validate_sections()

    def get_gridpoints_in_polygon(self, res_file: str | Path) -> list[int]:
        """Convenience method to get edges in section.

        This method is an overload of the parent method with fixed
        property_name and dtype arguments.

        Args:
            res_file (str | Path): path to result (map) netcdf file.

        Returns:
            list[str]: List of section ids for each edge in the grid.
        """
        super().get_gridpoints_in_polygon(res_file, "edge", "section")

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
        self.check_overlap()

        if raise_exception:
            err_msg = "Section file is not valid"
            raise OSError(err_msg)
        self.set_logger_message("Section file succesfully validated")
