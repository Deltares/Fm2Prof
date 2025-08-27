"""Contains code for importing files to fm2prof."""

from __future__ import annotations

# import from standard library
from pathlib import Path

# import from dependencies
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

# import from  package
from fm2prof.common import FM2ProfBase


class FMDataImporter(FM2ProfBase):
    """FM Data importer class."""

    def __init__(self, file_path: Path | str) -> None:
        """Initialize the FMDataImporter."""
        super().__init__()
        self.set_logger_message("FMDataImporter initialized", "debug")
        self.file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        """Return the file path."""
        return self._file_path

    @file_path.setter
    def file_path(self, value: Path | str) -> None:
        """Set the file path."""
        if isinstance(value, str):
            value = Path(value)
        if not value.exists():
            raise FileNotFoundError(f"The file {value} does not exist.")
        self._file_path = value

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
        """Mapping with dflow2d_result_keys."""
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
        """Get a variable from the netCDF file."""
        grid = xr.open_dataset(self.file_path, engine="netcdf4")
        return grid[var_name].to_numpy()

    def import_dflow2d(self) -> tuple[pd.DataFrame | None, dict, pd.DataFrame, dict]:
        """Read input from a dflow2d output file.

        Args:
            file_path: path to *_map.nc file

        Results:
            tid_face - DataFrame with time-independent data on faces (e.g. section allocation)
            tid_edge - DataFrame with time-independent data on flow links
            node_coordinates -
            td - DataFrame with time-dependent data (e.g. water levels, ..)

        """
        self.set_logger_message("hello from dflow2d importer")

        # Open results file for reading, within context manager to ensure garbage collection
        with Dataset(self.file_path, "r") as map_file:
            # Time-invariant variables from FM 2D at faces
            # -----------------------------------------------
            tid_face = None
            for key, nckey in self.dflow2d_face_keys.items():
                if tid_face is None:
                    tid_face = pd.DataFrame(columns=[key], data=np.array(map_file.variables[nckey]))
                else:
                    tid_face[key] = np.array(map_file.variables[nckey])

            # These variables are preallocated with the correct size for later use
            tid_face["region"] = [""] * len(tid_face["y"])  # region id (see RegionPolygon). By default, no regions
            tid_face["section"] = ["main"] * len(
                tid_face["y"],
            )  # section id (see SectionPolygon). By default, all sections are 'main'
            tid_face["sclass"] = [""] * len(tid_face["y"])  # cross-section id
            tid_face["islake"] = [False] * len(
                tid_face["y"],
            )  # whether or not cell is in a lake. By default, nothing is a lake

            # Time-invariant variables from FM 2D at edges
            # -----------------------------------------------
            internal_edges = map_file.variables["mesh2d_edge_type"][:] == 1  # edgetype 1 = 'internal'

            tid_edge = {}
            for key, nckey in self.dflow2d_edge_keys.items():
                try:
                    tid_edge[key] = np.array(map_file.variables[nckey])[internal_edges]
                except KeyError:
                    # 'edge_faces' does not always seem to exist in the file - probably
                    # due to changes in dflow2d output
                    self.set_logger_message(
                        f"during reading of dflow2d input, it was found that {key} was not present in the file",
                        "warning",
                    )

            tid_edge["sclass"] = np.array([""] * np.sum(internal_edges), dtype="U99")
            tid_edge["section"] = np.array(["main"] * np.sum(internal_edges), dtype="U99")
            tid_edge["region"] = np.array([""] * np.sum(internal_edges), dtype="U99")

            # node data (- Is this data still used??)
            # ----------------------------------------------
            node_coordinates = pd.DataFrame(columns=["x"], data=np.array(map_file.variables["mesh2d_node_x"]))
            node_coordinates["y"] = np.array(map_file.variables["mesh2d_node_y"])

            # Time-variant variables
            # ----------------------------------------------
            td = {}
            for key, nckey in self.dflow2d_result_keys.items():
                if key == "chezy_edge":
                    # this one we treat slightly differently:
                    # because is it edge_data, we need to filter on [internal_edges]
                    # also - older dflow2d versions (before december 2020) do not have
                    # the 'mesh2d_czu' keyword, so we need to default back to 'mesh2d_cftrt'
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


class FmModelData:
    """Used to read and store data from the 2D model."""

    time_dependent_data = None
    time_independent_data = None
    edge_data = None
    node_coordinates = None
    css_data_list = None

    def __init__(
        self,
        time_dependent_data: pd.DataFrame,
        time_independent_data: pd.DataFrame,
        edge_data: dict,
        node_coordinates: pd.DataFrame,
        css_data_dictionary: dict,
    ) -> None:
        """Instantiate a FmModelData object.

        Args:
            time_dependent_data (pd.DataFrame): _description_
            time_independent_data (pd.DataFrame): _description_
            edge_data (dict): _description_
            node_coordinates (pd.DataFrame): _description_
            css_data_dictionary (dict): _description_

        """
        self.time_dependent_data = time_dependent_data
        self.time_independent_data = time_independent_data
        self.edge_data = edge_data
        self.node_coordinates = node_coordinates
        self.css_data_list = self.get_ordered_css_list(css_data_dictionary)

    @staticmethod
    def get_ordered_css_list(css_data_dict: dict[str, str]) -> list[dict[str, str]]:
        """Return an ordered list where every element represents a Cross Section structure.

        Args:
            css_data_dict (dict[str, str]): Dictionary ordered by the keys

        Returns:
            (list): List where every element contains a dictionary
            to create a Cross Section.

        """
        if not css_data_dict or not isinstance(css_data_dict, dict):
            return []

        number_of_css = len(css_data_dict[next(iter(css_data_dict))])
        css_dict_keys = css_data_dict.keys()
        css_dict_values = css_data_dict.values()
        return [
            dict(
                zip(
                    css_dict_keys,
                    [value[idx] for value in css_dict_values if idx < len(value)],
                ),
            )
            for idx in range(number_of_css)
        ]

    def get_selection(self, css_name: str) -> dict:
        """Create a dictionary that holds all the 2D data for the cross-section with name 'css_name'.

        Args:
            css_name (str): name of the cross-section

        """
        dti = self.time_independent_data
        dtd = self.time_dependent_data
        edge_data = self.edge_data

        x = dti["x"][dti["sclass"] == css_name]
        y = dti["y"][dti["sclass"] == css_name]
        area = dti["area"][dti["sclass"] == css_name]
        region = dti["region"][dti["sclass"] == css_name]
        islake = dti["islake"][dti["sclass"] == css_name]
        waterdepth = dtd["waterdepth"][dti["sclass"] == css_name]
        waterlevel = dtd["waterlevel"][dti["sclass"] == css_name]
        vx = dtd["velocity_x"][dti["sclass"] == css_name]
        vy = dtd["velocity_y"][dti["sclass"] == css_name]
        face_section = dti["section"][dti["sclass"] == css_name]
        # find all chezy values for this cross section, note that edge coordinates are used
        chezy = dtd["chezy_edge"][edge_data["sclass"] == css_name]
        try:
            edge_faces = edge_data["edge_faces"][edge_data["sclass"] == css_name]
        except KeyError:
            edge_faces = None

        edge_x = edge_data["x"][edge_data["sclass"] == css_name]
        edge_y = edge_data["y"][edge_data["sclass"] == css_name]
        edge_section = edge_data["section"][edge_data["sclass"] == css_name]  # roughness section number

        bedlevel = dti["bedlevel"][dti["sclass"] == css_name]

        velocity = (vx**2 + vy**2) ** 0.5
        waterlevel[waterdepth == 0] = np.nan

        return {
            "x": x,
            "y": y,
            "area": area,
            "bedlevel": bedlevel,
            "waterdepth": waterdepth,
            "waterlevel": waterlevel,
            "velocity": velocity,
            "section": face_section,
            "chezy": chezy,
            "region": region,
            "islake": islake,
            "edge_faces": edge_faces,
            "edge_x": edge_x,
            "edge_y": edge_y,
            "edge_section": edge_section,
        }


class ImportInputFiles(FM2ProfBase):
    """Contains all functions related to the import of files."""

    def css_file(self, file_path: Path | str, delimiter: str = ",") -> dict:
        """Read the cross-section location file."""
        skip_line = False  # flag to skip line if file has header

        if not file_path or not Path(file_path).exists():
            err_msg = f"No file path for Cross Section location file was given, or could not be found at {file_path}"
            raise OSError(err_msg)

        with Path(file_path).open("r") as fid:
            input_data = {"xy": [], "id": [], "branchid": [], "length": [], "chainage": []}
            for lineno, line in enumerate(fid):
                try:
                    (cssid, x, y, length, branchid, chainage) = line.split(delimiter)
                except ValueError:
                    # revert to legacy format
                    (x, y, branchid, length, chainage) = line.split(delimiter)
                    cssid = branchid + "_" + str(round(float(chainage)))
                try:
                    float(x)
                except ValueError:
                    if lineno == 0:
                        # file has header. Skip header and try again next
                        skip_line = True
                if not skip_line:
                    input_data["xy"].append((float(x), float(y)))
                    input_data["id"].append(cssid)
                    input_data["length"].append(float(length))
                    input_data["branchid"].append(branchid.strip())
                    input_data["chainage"].append(float(chainage))
                skip_line = False

            # Convert everything to ndarray
            for key in input_data:
                input_data[key] = np.array(input_data[key])
            return input_data
