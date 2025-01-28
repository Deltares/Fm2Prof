import contextlib
import os
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

try:
    from pip import main as pipmain
except Exception:
    from pip._internal import main as pipmain


class TestUtils:
    _name_external = "external_test_data"
    _name_local = "test_data"
    _name_artifacts = "artifacts"
    _temp_copies = "temp-copies"

    @staticmethod
    def install_package(package: str):
        """Installs a package that is normally only used
        by a test configuration.

        Arguments:
            package {str} -- Name of the PIP package.
        """
        pipmain(["install", package])

    @staticmethod
    def get_external_test_data_dir() -> Path:
        """Gets the path to the external test data directory.

        Returns:
            Path: Directory path.
        """
        return Path(__file__).parent / TestUtils._name_external

    @staticmethod
    def get_external_test_data_subdir(subdir: str) -> Path:
        return TestUtils.get_external_test_data_dir() / subdir

    @staticmethod
    def get_artifacts_test_data_dir() -> Path:
        return Path(__file__).parent / TestUtils._name_artifacts

    @staticmethod
    def get_local_test_data_dir(dir_name: str) -> Path:
        """Returns the desired directory relative to the test data.
        Avoiding extra code on the tests.
        """
        directory = TestUtils.get_test_data_dir(dir_name, TestUtils._name_local)
        return directory

    @staticmethod
    def get_external_repo(dir_name: str) -> Path:
        """Returns the parent directory of this repo directory.

        Args:
            dir_name (str): Repo 'sibbling' of the current one.

        Returns:
            Path: Path to the sibbling repo.
        """
        return Path(__file__).parent.parent.parent / dir_name

    @staticmethod
    def get_test_data_dir(dir_name: str, test_data_name: str) -> Path:
        """Returns the desired directory relative to the test external data.
        Avoiding extra code on the tests.
        """
        return Path(__file__).parent / test_data_name / dir_name

    @staticmethod
    def get_local_test_file(filepath: str) -> Path:
        return Path(__file__).parent / TestUtils._name_local / filepath

    @staticmethod
    @contextlib.contextmanager
    def working_directory(path: Path):
        """Changes working directory and returns to previous on exit."""
        prev_cwd = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    @staticmethod
    def read_fm_model(file_path):
        """input: FM2D map file"""
        fm_edge_keys = {
            "x": "mesh2d_edge_x",
            "y": "mesh2d_edge_y",
            "edge_faces": "mesh2d_edge_faces",
            "edge_nodes": "mesh2d_edge_nodes",
        }
        edge_data = dict()
        # Open results file for reading
        res_fid = netCDF4.Dataset(file_path, "r")
        # Time-invariant variables from FM 2D
        df = pd.DataFrame(columns=["x"], data=np.array(res_fid.variables["mesh2d_face_x"]))
        df["y"] = np.array(res_fid.variables["mesh2d_face_y"])
        df["area"] = np.array(res_fid.variables["mesh2d_flowelem_ba"])
        df["bedlevel"] = np.array(res_fid.variables["mesh2d_flowelem_bl"])
        # These are filled later
        df["region"] = [""] * len(df["y"])
        df["section"] = ["main"] * len(df["y"])
        df["sclass"] = [""] * len(df["y"])  # cross-section id
        df["islake"] = [False] * len(df["y"])  # roughness section number

        # Edge data
        # edgetype 1 = 'internal'
        internal_edges = res_fid.variables["mesh2d_edge_type"][:] == 1
        for key, value in fm_edge_keys.items():
            try:
                edge_data[key] = np.array(res_fid.variables[value])[internal_edges]
            except KeyError:
                # 'edge_faces' does not always seem to exist in the file.
                # TODO: incorporate this function in its FmModelData with logger to
                # output a warning. For now, the omission of 'edge_faces' is handled
                # in FmModelData.
                pass
        edge_data["sclass"] = np.array([""] * np.sum(internal_edges), dtype="U99")
        edge_data["section"] = np.array(["main"] * np.sum(internal_edges), dtype="U99")
        edge_data["region"] = np.array([""] * np.sum(internal_edges), dtype="U99")
        # node data (not used?)
        df_node = pd.DataFrame(columns=["x"], data=np.array(res_fid.variables["mesh2d_node_x"]))
        df_node["y"] = np.array(res_fid.variables["mesh2d_node_y"])

        # Time-variant variables
        time_dependent = {
            "waterdepth": pd.DataFrame(
                data=np.array(res_fid.variables["mesh2d_waterdepth"]).T, columns=res_fid.variables["time"]
            ),
            "waterlevel": pd.DataFrame(
                data=np.array(res_fid.variables["mesh2d_s1"]).T, columns=res_fid.variables["time"]
            ),
            "chezy_mean": pd.DataFrame(
                data=np.array(res_fid.variables["mesh2d_czs"]).T, columns=res_fid.variables["time"]
            ),
            "chezy_edge": pd.DataFrame(
                data=np.array(res_fid.variables["mesh2d_cftrt"]).T[internal_edges], columns=res_fid.variables["time"]
            ),
            "velocity_x": pd.DataFrame(
                data=np.array(res_fid.variables["mesh2d_ucx"]).T, columns=res_fid.variables["time"]
            ),
            "velocity_y": pd.DataFrame(
                data=np.array(res_fid.variables["mesh2d_ucy"]).T, columns=res_fid.variables["time"]
            ),
            "velocity_edge": pd.DataFrame(
                data=np.array(res_fid.variables["mesh2d_u1"]).T, columns=res_fid.variables["time"]
            ),
        }
        return df, edge_data, df_node, time_dependent


skipwhenexternalsmissing = pytest.mark.skipif(
    not (TestUtils.get_external_test_data_dir().is_dir()),
    reason="Only to be run to generate expected data from local machines.",
)
