"""Base importer class and generalized ModelData for all import formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from fm2prof.common import FM2ProfBase


def _ndarray_setattr(obj: object, name: str, value: object) -> None:
    """Coerce value to np.ndarray using dtype declared in field metadata."""
    for f in fields(obj.__class__):
        if f.name == name:
            dtype = f.metadata.get("dtype")
            if dtype is not None:
                value = np.asarray(value, dtype=dtype)
            break
    object.__setattr__(obj, name, value)


@dataclass
class FaceGeometry:
    """Geometric properties of 2D mesh faces (cells). All arrays have length N_faces."""

    x:        np.ndarray = field(metadata={"dtype": float})   # face centroid x-coordinate [m]
    y:        np.ndarray = field(metadata={"dtype": float})   # face centroid y-coordinate [m]
    area:     np.ndarray = field(metadata={"dtype": float})   # face area [m2]
    bedlevel: np.ndarray = field(metadata={"dtype": float})   # bed level [m+NAP]
    section:  np.ndarray = field(metadata={"dtype": object})  # section classification (main/floodplain)
    region:   np.ndarray = field(metadata={"dtype": object})  # region label → cross-section name
    islake:   np.ndarray = field(metadata={"dtype": bool})    # True if face belongs to a lake
    sclass:   np.ndarray = field(metadata={"dtype": object})  # cross-section class label for selection

    def __setattr__(self, name: str, value: object) -> None:
        _ndarray_setattr(self, name, value)


@dataclass
class EdgeGeometry:
    """Geometric properties of 2D mesh edges (flow links). All arrays have length N_edges."""

    x:          np.ndarray = field(metadata={"dtype": float})   # edge centroid x-coordinate [m]
    y:          np.ndarray = field(metadata={"dtype": float})   # edge centroid y-coordinate [m]
    section:    np.ndarray = field(metadata={"dtype": object})  # section classification per edge
    region:     np.ndarray = field(metadata={"dtype": object})  # region classification per edge
    sclass:     np.ndarray = field(metadata={"dtype": object})  # cross-section class label for selection
    edge_nodes: np.ndarray = field(metadata={"dtype": int})     # node indices per edge, shape (N_edges, 2)
    edge_faces: np.ndarray | None = None  # face indices per edge, shape (N_edges, 2)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "edge_faces":
            object.__setattr__(self, name, np.asarray(value) if value is not None else None)
        else:
            _ndarray_setattr(self, name, value)


@dataclass
class HydraulicData:
    """Time-dependent hydraulic results. Shape (N_timesteps, N_faces) unless noted."""

    waterlevel: np.ndarray = field(metadata={"dtype": float})  # water surface level [m+NAP]
    waterdepth: np.ndarray = field(metadata={"dtype": float})  # water depth [m]
    velocity_x: np.ndarray = field(metadata={"dtype": float})  # x-component depth-averaged velocity [m/s]
    velocity_y: np.ndarray = field(metadata={"dtype": float})  # y-component depth-averaged velocity [m/s]
    chezy_edge: np.ndarray = field(metadata={"dtype": float})  # Chezy roughness on edges [m0.5/s]

    def __setattr__(self, name: str, value: object) -> None:
        _ndarray_setattr(self, name, value)


@dataclass
class CrossSectionData:
    """Definition of a single cross-section location from the 1D model."""

    name: str
    """Unique cross-section identifier."""

    length: float
    """Representative length [m]."""

    location: tuple[float, float]
    """(x, y) coordinates of the cross-section."""

    branch_id: str
    """Branch identifier in the 1D network."""

    offset: float
    """Offset along the branch [m]."""


class ModelData:
    """Source-agnostic container for all data required to generate 1D cross-sections.

    ``geometry`` is always required. ``edges`` and ``hydraulics`` are optional
    to support future use cases where only geometric data is available
    (e.g. mesh inspection, dry-run validation).

    Attributes:
        geometry:       Required. Geometric properties of 2D mesh faces.
        cross_sections: Required. Ordered list of cross-section definitions.
        source:         Required. Format identifier, e.g. ``'dflowfm'``.
        edges:          Optional. Geometric properties of 2D mesh edges.
        hydraulics:     Optional. Time-dependent hydraulic results.

    """

    def __init__(
        self,
        geometry: FaceGeometry,
        cross_sections: list[CrossSectionData],
        source: str,
        edges: EdgeGeometry | None = None,
        hydraulics: HydraulicData | None = None,
    ) -> None:
        self.geometry = geometry
        self.cross_sections = cross_sections
        self.source = source
        self.edges = edges
        self.hydraulics = hydraulics
        self.css_data_list: list[dict] = []

    @property
    def has_hydraulics(self) -> bool:
        """Return True if hydraulic data is available."""
        return self.hydraulics is not None

    @property
    def has_edges(self) -> bool:
        """Return True if edge geometry data is available."""
        return self.edges is not None

    def get_selection(self, css_name: str) -> dict:
        """Return all 2D data for cross-section ``css_name``.

        Args:
            css_name: Name of the cross-section to select.

        Returns:
            Dictionary with all available 2D data for the cross-section.
            Keys for edges and hydraulics are omitted if those are not available.

        Raises:
            ValueError: If hydraulics are required but not available.

        """
        g = self.geometry
        mask_face = g.sclass == css_name
        face_idx = np.nonzero(mask_face)[0]

        result = {
            "x":        pd.Series(g.x[mask_face],        index=face_idx),
            "y":        pd.Series(g.y[mask_face],        index=face_idx),
            "area":     pd.Series(g.area[mask_face],     index=face_idx),
            "bedlevel": pd.Series(g.bedlevel[mask_face], index=face_idx),
            "section":  pd.Series(g.section[mask_face],  index=face_idx),
            "region":   pd.Series(g.region[mask_face],   index=face_idx),
            "islake":   pd.Series(g.islake[mask_face],   index=face_idx),
        }

        if self.has_hydraulics:
            h = self.hydraulics
            waterdepth = pd.DataFrame(h.waterdepth[mask_face], index=face_idx)
            waterlevel = pd.DataFrame(h.waterlevel[mask_face].copy(), index=face_idx)
            waterlevel[waterdepth == 0] = np.nan
            vx = h.velocity_x[mask_face]
            vy = h.velocity_y[mask_face]

            result.update({
                "waterdepth": waterdepth,
                "waterlevel": waterlevel,
                "velocity":   pd.DataFrame((vx**2 + vy**2) ** 0.5, index=face_idx),
            })

        if self.has_edges:
            e = self.edges
            mask_edge = e.sclass == css_name
            result.update({
                "edge_x":       e.x[mask_edge],
                "edge_y":       e.y[mask_edge],
                "edge_section": e.section[mask_edge],
                "edge_faces":   e.edge_faces[mask_edge] if e.edge_faces is not None else None,
            })

        if self.has_hydraulics and self.has_edges:
            mask_edge = self.edges.sclass == css_name
            result["chezy"] = pd.DataFrame(self.hydraulics.chezy_edge[mask_edge])

        return result

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
