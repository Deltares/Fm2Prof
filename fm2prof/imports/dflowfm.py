"""DFlowFM-specific importer, migrated from data_import.py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

from fm2prof.imports.base import BaseImporter, ModelData

if TYPE_CHECKING:
    pass


class DFlowFMImporter(BaseImporter):
    """Importer for D-Flow FM 2D map output files (*_map.nc).

    Migrated from FMDataImporter in fm2prof.data_import.
    """

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
        grid = xr.open_dataset(self.file_path, engine="netcdf4")
        return grid[var_name].to_numpy()

    def import_data(self) -> ModelData:
        """Import data from a D-Flow FM map file and return a ModelData object.

        Returns:
            ModelData object containing all imported data.

        """
        tid_face, tid_edge, node_coordinates, td = self._import_dflow2d()
        return ModelData(
            time_dependent_data=td,
            time_independent_data=tid_face,
            edge_data=tid_edge,
            node_coordinates=node_coordinates,
            css_data_dictionary={},
            source=self.SOURCE,
        )

    def _import_dflow2d(self) -> tuple[pd.DataFrame | None, dict, pd.DataFrame, dict]:
        """Read input from a dflow2d output file.

        Returns:
            tid_face: DataFrame with time-independent data on faces (e.g. section allocation).
            tid_edge: Dictionary with time-independent data on flow links.
            node_coordinates: DataFrame with node coordinates.
            td: Dictionary with time-dependent data (e.g. water levels).

        """
        self.set_logger_message("hello from dflow2d importer")

        with Dataset(self.file_path, "r") as map_file:
            # Time-invariant variables from FM 2D at faces
            tid_face = None
            for key, nckey in self.dflow2d_face_keys.items():
                if tid_face is None:
                    tid_face = pd.DataFrame(columns=[key], data=np.array(map_file.variables[nckey]))
                else:
                    tid_face[key] = np.array(map_file.variables[nckey])

            tid_face["region"] = [""] * len(tid_face["y"])
            tid_face["section"] = ["main"] * len(tid_face["y"])
            tid_face["sclass"] = [""] * len(tid_face["y"])
            tid_face["islake"] = [False] * len(tid_face["y"])

            # Time-invariant variables from FM 2D at edges
            internal_edges = map_file.variables["mesh2d_edge_type"][:] == 1

            tid_edge = {}
            for key, nckey in self.dflow2d_edge_keys.items():
                try:
                    tid_edge[key] = np.array(map_file.variables[nckey])[internal_edges]
                except KeyError:
                    self.set_logger_message(
                        f"during reading of dflow2d input, it was found that {key} was not present in the file",
                        "warning",
                    )

            tid_edge["sclass"] = np.array([""] * np.sum(internal_edges), dtype="U99")
            tid_edge["section"] = np.array(["main"] * np.sum(internal_edges), dtype="U99")
            tid_edge["region"] = np.array([""] * np.sum(internal_edges), dtype="U99")

            # Node coordinates
            node_coordinates = pd.DataFrame(columns=["x"], data=np.array(map_file.variables["mesh2d_node_x"]))
            node_coordinates["y"] = np.array(map_file.variables["mesh2d_node_y"])

            # Time-variant variables
            td = {}
            for key, nckey in self.dflow2d_result_keys.items():
                if key == "chezy_edge":
                    try:
                        td[key] = pd.DataFrame(
                            data=np.array(map_file.variables[nckey]).T[internal_edges],
                            columns=map_file.variables["time"],
                        )
                    except KeyError:
                        td[key] = pd.DataFrame(
                            data=np.array(map_file.variables["mesh2d_cftrt"]).T[internal_edges],
                            columns=map_file.variables["time"],
                        )
                        self.set_logger_message(
                            "The Dflow2D output does not have the 'mesh2d_czu' key. Reverting to mesh2d_cftrt. "
                            "Make sure that the UnifFrictType is set to 0 (Cheyz) in the Dflow2d mdu file.",
                            "warning",
                        )
                else:
                    td[key] = pd.DataFrame(
                        data=np.array(map_file.variables[nckey]).T,
                        columns=map_file.variables["time"],
                    )

        return tid_face, tid_edge, node_coordinates, td
