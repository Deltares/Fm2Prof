"""DFlowFM-specific importer, migrated from data_import.py."""

from __future__ import annotations

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from fm2prof.imports.base import (
    BaseImporter,
    EdgeGeometry,
    FaceGeometry,
    HydraulicData,
    ModelData,
)


class DFlowFMImporter(BaseImporter):
    """Importer for D-Flow FM 2D map output files (*_map.nc)."""

    SOURCE = "dflowfm"

    @property
    def dflow2d_face_keys(self) -> dict:
        """Mapping with dflow2d face keys."""
        return {
            "x": "mesh2d_face_x",
            "y": "mesh2d_face_y",
            "area": "mesh2d_flowelem_ba",
            "bedlevel": "mesh2d_flowelem_bl",
        }

    @property
    def dflow2d_edge_keys(self) -> dict:
        """Mapping with dflow2d edge keys."""
        return {
            "x": "mesh2d_edge_x",
            "y": "mesh2d_edge_y",
            "edge_faces": "mesh2d_edge_faces",
            "edge_nodes": "mesh2d_edge_nodes",
        }

    @property
    def dflow2d_result_keys(self) -> dict:
        """Mapping with dflow2d result keys."""
        return {
            "waterdepth": "mesh2d_waterdepth",
            "waterlevel": "mesh2d_s1",
            "chezy_mean": "mesh2d_czs",  # not used anymore!
            "chezy_edge": "mesh2d_czu",
            "velocity_x": "mesh2d_ucx",
            "velocity_y": "mesh2d_ucy",
            "velocity_edge": "mesh2d_u1",
        }

    def get_variable(self, var_name: str) -> np.ndarray:
        """Get a variable from the netCDF file.

        Args:
            var_name: Name of the variable to retrieve.

        Returns:
            Variable data as numpy array.

        """
        with xr.open_dataset(self.file_path, engine="netcdf4") as grid:
            return grid[var_name].to_numpy()

    def import_data(self) -> ModelData:
        """Import data from a D-Flow FM map file and return a ModelData object.

        Returns:
            ModelData with geometry, edges, and hydraulics populated.
            cross_sections is empty by default; it is populated later
            by fm2prof_runner after reading the css location file.

        """
        self.set_logger_message("Reading D-Flow FM map file")

        with Dataset(self.file_path, "r") as map_file:
            internal_edges = map_file.variables["mesh2d_edge_type"][:] == 1
            geometry = self._read_face_geometry(map_file)
            edges = self._read_edge_geometry(map_file, internal_edges)
            hydraulics = self._read_hydraulic_data(map_file, internal_edges)

        return ModelData(
            geometry=geometry,
            cross_sections=[],
            source=self.SOURCE,
            edges=edges,
            hydraulics=hydraulics,
        )

    # --- Private helpers ------------------------------------------------------

    def _read_face_geometry(self, map_file: Dataset) -> FaceGeometry:
        """Read time-invariant face geometry from the map file."""
        n_faces = len(np.array(map_file.variables["mesh2d_face_x"]))
        return FaceGeometry(
            x=        np.array(map_file.variables["mesh2d_face_x"]),
            y=        np.array(map_file.variables["mesh2d_face_y"]),
            area=     np.array(map_file.variables["mesh2d_flowelem_ba"]),
            bedlevel= np.array(map_file.variables["mesh2d_flowelem_bl"]),
            # classification fields — populated later by fm2prof_runner
            section=  np.array(["main"] * n_faces, dtype="object"),
            region=   np.array([""]     * n_faces, dtype="object"),
            sclass=   np.array([""]     * n_faces, dtype="object"),
            islake=   np.zeros(n_faces, dtype=bool),
        )

    def _read_edge_geometry(self, map_file: Dataset, internal_edges: np.ndarray) -> EdgeGeometry:
        """Read time-invariant edge geometry from the map file."""
        n_edges = int(np.sum(internal_edges))

        edge_faces = None
        try:
            edge_faces = np.array(map_file.variables["mesh2d_edge_faces"])[internal_edges]
        except KeyError:
            self.set_logger_message(
                "mesh2d_edge_faces not present in the file - edge_faces will be None.",
                "warning",
            )

        try:
            edge_nodes = np.array(map_file.variables["mesh2d_edge_nodes"])[internal_edges]
        except KeyError:
            self.set_logger_message(
                "mesh2d_edge_nodes not present in the file - edge_nodes will be empty.",
                "warning",
            )
            edge_nodes = np.empty((n_edges, 2), dtype=int)

        return EdgeGeometry(
            x =          np.array(map_file.variables["mesh2d_edge_x"])[internal_edges],
            y =          np.array(map_file.variables["mesh2d_edge_y"])[internal_edges],
            edge_nodes = edge_nodes,
            edge_faces = edge_faces,
            # classification fields — populated later by fm2prof_runner
            section =    np.array(["main"] * n_edges, dtype="U99"),
            region =    np.array(["undefined"] * n_edges, dtype="U99"),
            sclass =     np.array([""]     * n_edges, dtype="U99"),
        )

    def _read_hydraulic_data(self, map_file: Dataset, internal_edges: np.ndarray) -> HydraulicData:
        """Read time-dependent hydraulic results from the map file."""
        def _face_array(nckey: str) -> np.ndarray:
            return np.array(map_file.variables[nckey]).T

        def _edge_array(nckey: str) -> np.ndarray:
            return np.array(map_file.variables[nckey]).T[internal_edges]

        try:
            chezy_edge = _edge_array("mesh2d_czu")
        except KeyError:
            chezy_edge = _edge_array("mesh2d_cftrt")
            self.set_logger_message(
                "The D-Flow FM output does not have the 'mesh2d_czu' key. Reverting to mesh2d_cftrt. "
                "Make sure that UnifFrictType is set to 0 (Chezy) in the D-Flow FM mdu file.",
                "warning",
            )

        return HydraulicData(
            waterlevel= _face_array("mesh2d_s1"),
            waterdepth= _face_array("mesh2d_waterdepth"),
            velocity_x= _face_array("mesh2d_ucx"),
            velocity_y= _face_array("mesh2d_ucy"),
            chezy_edge= chezy_edge,
        )
