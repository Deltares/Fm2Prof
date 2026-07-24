"""Base importer class and generalized ModelData for all import formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fm2prof.common import FM2ProfBase

if TYPE_CHECKING:
    pass


class ModelData:
    """Generalized model data container, source-agnostic.

    Stores all data read from a 2D model, regardless of the source format.
    Replaces the format-specific FmModelData class.
    """

    time_dependent_data: dict | None = None
    time_independent_data: pd.DataFrame | None = None
    edge_data: dict | None = None
    node_coordinates: pd.DataFrame | None = None
    css_data_list: list | None = None
    source: str = ""

    def __init__(
        self,
        time_dependent_data: dict,
        time_independent_data: pd.DataFrame,
        edge_data: dict,
        node_coordinates: pd.DataFrame,
        css_data_dictionary: dict,
        source: str = "",
    ) -> None:
        """Instantiate a ModelData object.

        Args:
            time_dependent_data: Time-dependent data (e.g. water levels, velocities).
            time_independent_data: Time-independent data on faces (e.g. section allocation).
            edge_data: Time-independent data on flow links.
            node_coordinates: Node coordinates.
            css_data_dictionary: Cross-section data dictionary.
            source: Identifier of the source format (e.g. 'dflowfm').

        """
        self.time_dependent_data = time_dependent_data
        self.time_independent_data = time_independent_data
        self.edge_data = edge_data
        self.node_coordinates = node_coordinates
        self.css_data_list = self.get_ordered_css_list(css_data_dictionary)
        self.source = source

    @staticmethod
    def get_ordered_css_list(css_data_dict: dict[str, str]) -> list[dict[str, str]]:
        """Return an ordered list where every element represents a Cross Section structure.

        Args:
            css_data_dict: Dictionary ordered by the keys.

        Returns:
            List where every element contains a dictionary to create a Cross Section.

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
            css_name: Name of the cross-section.

        Returns:
            Dictionary with all 2D data for the given cross-section.

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
        chezy = dtd["chezy_edge"][edge_data["sclass"] == css_name]

        try:
            edge_faces = edge_data["edge_faces"][edge_data["sclass"] == css_name]
        except KeyError:
            edge_faces = None

        edge_x = edge_data["x"][edge_data["sclass"] == css_name]
        edge_y = edge_data["y"][edge_data["sclass"] == css_name]
        edge_section = np.array(edge_data["section"])[edge_data["sclass"] == css_name]

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


class BaseImporter(FM2ProfBase, ABC):
    """Abstract base class for all format-specific importers.

    All importers must implement the import_data method, which returns
    a ModelData object regardless of the source format.
    """

    def __init__(self, file_path: Path | str) -> None:
        """Initialize the importer.

        Args:
            file_path: Path to the input file.

        """
        super().__init__()
        self.file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        """Return the file path."""
        return self._file_path

    @file_path.setter
    def file_path(self, value: Path | str) -> None:
        """Set and validate the file path."""
        if isinstance(value, str):
            value = Path(value)
        if not value.exists():
            raise FileNotFoundError(f"The file {value} does not exist.")
        self._file_path = value

    @abstractmethod
    def import_data(self) -> ModelData:
        """Import data and return a ModelData object.

        Returns:
            ModelData object containing all imported data.

        """
        ...
