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

import shapely
from meshkernel import GeometryList, MeshKernel, ProjectionType

from fm2prof.common import FM2ProfBase
from fm2prof.data_import import FMDataImporter

if TYPE_CHECKING:
    from logging import Logger

    import numpy as np

class PolygonError(Exception):
    """Custom exception for polygon errors."""
    def __init__(self, message: str="Polygon file is not valid") -> None:
        """Initialize PolygonError with a message."""
        self.message = message
        super().__init__(self.message)

class GridPointsInPolygonResults(NamedTuple):
    """Named tuple for grid points in polygon results."""

    faces_in_polygon: list[str]
    edges_in_polygon: list[str]

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

    Not all geojson geometry types are supported:

    - Only 'Polygon' and 'MultiPolygon' are supported.
    - Polygons with holes are not supported.
    - MultiPolygons with multiple polygons are not supported.
    - Properties must contain a 'name' key-word.
    - Properties must be unique, otherwise overlap checking will produce bugs.
    - SectionPolygon properties must contain a 'section' key-word.
    - RegionPolygon properties must contain a 'region' key-word.

    """

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
        # Check if properties contain the required 'region' property
        regions = [polygon.properties.get("name") for polygon in polygons_list]
        if not all(regions):
            err_msg = "Polygon properties must contain key-word 'name'"
            raise ValueError(err_msg)
        # Check if 'region' property is unique, otherwise _check_overlap will produce bugs
        if len(regions) != len(set(regions)):
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
        res_file: str | Path,
        *,
        property_name: Literal["region", "section"],
        force_cache_invalidation: bool = False) -> GridPointsInPolygonResults:
        """Method to get faces and edges in region.

        This method performs caching of the in-polygon classification results
        to avoid recalculating if the region file has not changed. The cache
        is invalidated if the region file is modified or if `force_cache_invalidation`
        is set to `True`.

        Args:
            res_file (str | Path): path to result (map) netcdf file.
            property_name (Literal["region", "section"]): Property to use for classification.
            force_cache_invalidation (bool): Force cache invalidation even if region file has not changed.

        Returns:
            GridPointsInPolygonResults
        """
        # get metadata of file path to determine if the in-polygon classification needs
        # to be rerun
        meta = {"last_modified": res_file.stat().st_mtime,
                "file_size": res_file.stat().st_size}

        # check if cache file exists and is valid
        cache_file = Path(res_file).with_suffix(f".{property_name}_cache.json")
        if cache_file.exists() and not force_cache_invalidation:
            # read cache file
            meta_cache, faces_in_polygon, edges_in_polygon = self._load_cache(cache_file)
            if meta == meta_cache:
                self.set_logger_message("Using cached regions", level="info")
                return GridPointsInPolygonResults(faces_in_polygon=faces_in_polygon, edges_in_polygon=edges_in_polygon)
            self.set_logger_message("Cached regions are stale", level="warning")
            faces_in_polygon = self.meshkernel_inpolygon(res_file, dtype="face", property_name=property_name)
            edges_in_polygon = self.meshkernel_inpolygon(res_file, dtype="edge", property_name=property_name)
            self._write_cache(cache_file, meta, faces_in_polygon, edges_in_polygon)
        elif cache_file.exists() and force_cache_invalidation:
            self.set_logger_message("Forcing recalculating region cache", level="info")
            cache_file.unlink()
            faces_in_polygon = self.meshkernel_inpolygon(res_file, dtype="face", property_name=property_name)
            edges_in_polygon = self.meshkernel_inpolygon(res_file, dtype="edge", property_name=property_name)
            self._write_cache(cache_file, meta, faces_in_polygon, edges_in_polygon)
        elif not cache_file.exists():
            faces_in_polygon = self.meshkernel_inpolygon(res_file, dtype="face", property_name=property_name)
            edges_in_polygon = self.meshkernel_inpolygon(res_file, dtype="edge", property_name=property_name)
            self._write_cache(cache_file, meta, faces_in_polygon, edges_in_polygon)

        return GridPointsInPolygonResults(faces_in_polygon=faces_in_polygon, edges_in_polygon=edges_in_polygon)

    def get_points_in_polygon(self, points: np.ndarray, property_name: Literal["region", "section"]) -> list[str]:
        """Method to determine in which polygon input points are.

        Warning:
            This method is not applicable for large number of points.
            Only use for small number of points (e.g. cross-section locations).

        Args:
            points (np.ndarray): Array of shape (n_points, 2) containing x,y coordinates of points to classify.
            property_name (Literal['region', 'section']): Property to use for classification.

        Returns:
            list[str]: List of polygon names in which the points are located. If a point is not located
                       in any polygon, it is classified as 'undefined'.
        """
        # Convert to shapely point
        points = [shapely.Point(xy) for xy in points]
        points_regions = [self.undefined] * len(points)

        # Assign point to region
        for i, point in enumerate(points):
            for j, polygon in enumerate(self.as_shapely()):
                if point.within(polygon):
                    points_regions[i] = self.polygons[j].properties.get(property_name)
                    break

        return points_regions

    @staticmethod
    def _load_cache(cache_file: Path) -> tuple[dict, list[str], list[str]]:
        with cache_file.open("r") as f:
            cache_data = json.load(f)
        return cache_data.get("meta"), cache_data.get("faces"), cache_data.get("edges")

    @staticmethod
    def _write_cache(
        cache_file: Path,
        meta: dict,
        faces_in_region: list[str],
        edges_in_region: list[str]) -> None:
        cache_data = {
            "meta": meta,
            "faces": faces_in_region,
            "edges": edges_in_region}
        with cache_file.open("w") as f:
            json.dump(cache_data, f)

    def meshkernel_inpolygon(self,
        res_file: str | Path,  # path to result (map) netcdf file
        dtype: Literal["face", "edge", "node"],
        property_name: Literal["region", "section"],
        ) -> list[str]:
        """Get grid points in polygon.

        Args:
            res_file (str | Path): Path to result (map) netcdf file.
            dtype (Literal["face", "edge", "node"]): Type of grid points to retrieve.
            property_name (Literal["region", "section"]): Property to use for classification.

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
            # only internal edges!
            internal_edges = fmdata.get_variable("mesh2d_edge_type")[:] == 1
            edge_map: np.ndarray = fmdata.get_variable("mesh2d_edge_nodes").T[0] -1
            edge_map = edge_map[internal_edges]
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
                       f"{self.polygons[i].properties.get('name')} overlaps {self.polygons[j].properties.get('name')}.",
                       level="warning",
                    )

    def from_file(self, file_path: Path | str) -> None:
        """Read data from geojson file.

        Args:
            file_path (Path | str): path to geojson file
        """
        self._validate_extension(file_path)

        if Path(file_path).exists() is False:
            err_msg = "Polygon file does not exist"
            raise FileNotFoundError(err_msg)

        with Path(file_path).open("r") as geojson_file:
            try:
                geojson_data = json.load(geojson_file).get("features")
            except json.JSONDecodeError as e:
                err_msg = f"Error decoding JSON from {file_path}: {e}"
                raise PolygonError(err_msg) from e

        if not geojson_data:
            err_msg = "Polygon file has no features"
            raise PolygonError(err_msg)

        polygons: list[Polygon] = []
        for feature in geojson_data:
            feature_props = {k.lower(): v for k, v in feature.get("properties").items()}

            if "name" not in feature_props:
                err_msg = "Polygon properties must contain key-word 'name'"
                raise PolygonError(err_msg)

            geometry_type = feature["geometry"]["type"]
            if geometry_type not in ("Polygon", "MultiPolygon"):
                err_msg = "Polygon geometry must be of type 'Polygon' or 'MultiPolygon'"
                raise PolygonError(err_msg)

            geometry_coordinates = feature["geometry"]["coordinates"]
            if geometry_type == "Polygon":
                if len(geometry_coordinates) != 1:
                    err_msg = "Polygon geometry must contain a single polygon and no holes"
                    raise PolygonError(err_msg)
                geometry_coordinates = geometry_coordinates[0]
            elif geometry_type == "MultiPolygon":
                if len(geometry_coordinates) != 1 & len(geometry_coordinates[0]) != 1:
                    err_msg = "MultiPolygon geometry must contain a single polygon and no holes"
                    raise PolygonError(err_msg)
                geometry_coordinates = geometry_coordinates[0][0]
            polygons.append(Polygon(coordinates=geometry_coordinates, properties=feature_props))

        self.polygons = polygons

    @staticmethod
    def _validate_extension(file_path: Path | str) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix not in (".json", ".geojson"):
            err_msg = "Invalid file path extension, should be .json or .geojson."
            raise OSError(err_msg)


class RegionPolygon(MultiPolygon):
    """RegionPolygonFile class."""

    def __init__(self, region_file_path: str | Path, logger: Logger) -> None:
        """Instantiate a RegionPolygonFile object."""
        super().__init__(logger)
        try:
            self.from_file(region_file_path)
        except TypeError as e:
            self.set_logger_message(f"Potentially invalid geojson file: {e}", level="error")
            raise PolygonError from e

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

    def get_gridpoints_in_polygon(self,
        res_file: str | Path,
        *,
        property_name: Literal["region", "section"]="region",
        force_cache_invalidation:bool=False) -> GridPointsInPolygonResults:
        """Method to get faces and edges in region.

        This method performs caching of the in-polygon classification results
        to avoid recalculating if the region file has not changed. The cache
        is invalidated if the region file is modified or if `force_cache_invalidation`
        is set to `True`.

        Args:
            res_file (str | Path): path to result (map) netcdf file.
            property_name (Literal["region", "section"]): Property to use for classification. Defaults to region
            force_cache_invalidation (bool): Force cache invalidation even if region file has not changed.

        Returns:
            GridPointsInPolygonResults
        """
        return super().get_gridpoints_in_polygon(res_file,
                                              property_name=property_name,
                                              force_cache_invalidation=force_cache_invalidation)

    def _validate_regions(self) -> None:
        self.set_logger_message("Validating region file", level="info")

        number_of_regions = len(self.regions)

        self.set_logger_message(f"{number_of_regions} regions found", level="info")

        # Test if polygons overlap
        self.check_overlap()


class SectionPolygon(MultiPolygon):
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

    def get_gridpoints_in_polygon(self,
        res_file: str | Path,
        *,
        property_name: Literal["region", "section"]="section",
        force_cache_invalidation:bool=False) -> GridPointsInPolygonResults:
        """Method to get faces and edges in section.

        This method performs caching of the in-polygon classification results
        to avoid recalculating if the section file has not changed. The cache
        is invalidated if the section file is modified or if `force_cache_invalidation`
        is set to `True`.

        Args:
            res_file (str | Path): path to result (map) netcdf file.
            property_name (Literal["region", "section"]): Property to use for classification. Defaults to section
            force_cache_invalidation (bool): Force cache invalidation even if section file has not changed.

        Returns:
            GridPointsInPolygonResults
        """
        return super().get_gridpoints_in_polygon(res_file,
                                              property_name=property_name,
                                              force_cache_invalidation=force_cache_invalidation)

    def _validate_sections(self) -> None:
        self.set_logger_message("Validating Section file")
        raise_exception = False

        valid_section_keys = {"main", "floodplain1", "floodplain2", "1", "2", "3"}
        map_section_keys = {
            "1": "main",
            "2": "floodplain1",
            "3": "floodplain2",
        }

        # check section polygon validity
        for section in self.sections:
            if "section" not in section.properties:
                err_msg = f"Polygon {section.properties.get('name')} has no property 'section'"
                self.set_logger_message(
                    err_msg,
                    level="error",
                )
                raise PolygonError(err_msg)

            section_key = section.properties.get("section").lower()
            if section_key not in valid_section_keys:
                err_msg = f"{section_key} is not a recognized section"
                self.set_logger_message(
                    err_msg,
                    level="error",
                )
                raise PolygonError(err_msg)

            # remap 1, 2, 3 to main, floodplain1, floodplain2
            if section_key in map_section_keys:
                self.set_logger_message(
                    f"remapped section {section_key} to {map_section_keys[section_key]}",
                    level="warning",
                )
                section.properties["section"] = map_section_keys.get(section_key)

        # check for overlap (only raise a warning)
        self.check_overlap()

        if raise_exception:
            err_msg = "Section file is not valid"
            raise PolygonError(err_msg)
        self.set_logger_message("Section file succesfully validated")
