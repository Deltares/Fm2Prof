"""Data preprocessing pipeline for FM2PROF.

Provides a single entry point for loading, classifying and assembling
all 2D model data into a :class:`ModelData` instance that is ready for
cross-section generation.

The main function :func:`build_model_data` is source-agnostic: the
source format is selected via the ``source`` argument and resolved
through :class:`~fm2prof.imports.ImporterFactory`.

Functions:
    build_model_data: Load and classify all input data into a ModelData instance.
    classify_sections_by_variance: Classify faces/edges into roughness sections
        using variance-reduction on Chézy values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fm2prof import nearest_neighbour
from fm2prof.data_import import ImportInputFiles
from fm2prof.imports import ImporterFactory
from fm2prof.polygon_file import GridPointsInPolygonResults, RegionPolygon, SectionPolygon

if TYPE_CHECKING:
    import logging
    from logging import Logger
    from pathlib import Path

    from fm2prof.imports.base import ModelData
    from fm2prof.ini_file import InputFiles


def build_model_data(
    input_files: InputFiles,
    source: str = "dflowfm",
    *,
    default_region: str = "",
    default_section: str = "main",
    logger: logging.Logger | None = None,
) -> ModelData:
    """Load, classify and assemble all 2D model data into a ModelData instance.

    This function is the single entry point for the initialisation pipeline.
    It is source-agnostic: pass ``source="dflowfm"`` (or any supported source)
    and the appropriate importer is selected automatically.

    Steps performed:
        1. Import 2D map file via :class:`~fm2prof.imports.ImporterFactory`.
        2. Read cross-section location file.
        3. Optionally read region and section polygon files.
        4. Classify 2D faces and edges to cross-sections (nearest neighbour).
        5. Classify 2D faces and edges to roughness sections (polygon or variance).

    Args:
        input_files:     Container with paths to all required input files.
        source:          Source format identifier. Default: ``"dflowfm"``.
        default_region:  Default region label when no region polygon is given.
        default_section: Default section label when no section polygon is given.
        logger:          Optional logger instance for status messages.

    Returns:
        A fully classified :class:`~fm2prof.imports.ModelData` instance.

    """

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)

    # Ensure a valid logger is available for FM2ProfBase subclasses
    from fm2prof.common import FM2ProfBase  # noqa: PLC0415
    _base = FM2ProfBase()
    _base.set_logger(logger if logger is not None else _base.create_logger())
    _logger = _base.get_logger()

    # 1. Read input files
    model_data, cssdata, regions, sections = _read_input_files(
        input_files, source, default_region, default_section, _logger,
    )

    # 2 Classify faces and edges to cross-sections
    if regions is None:
        _log("Classifying 2D points to cross-sections (no region polygon)")
        model_data = _classify_cross_sections_without_regions(model_data, cssdata)
    else:
        _log("Classifying 2D points to cross-sections using region polygon")
        model_data = _classify_cross_sections_using_regions(model_data, cssdata, regions, input_files)

    # 3. Classify sections
    model_data = _classify_to_sections(model_data, sections, input_files.map_file)

    # 4. Attach cross-section definitions toi the ModelData object
    if cssdata and isinstance(cssdata, dict):
        n = len(cssdata[next(iter(cssdata))])
        keys = cssdata.keys()
        model_data.css_data_list = [
            dict(zip(keys, [v[i] for v in cssdata.values()], strict=False))
            for i in range(n)
        ]
    else:
        model_data.css_data_list = []

    return model_data

def _read_input_files(
    input_files: InputFiles,
    source: str,
    default_region: str,
    default_section: str,
    logger: Logger,
) -> tuple:
    """Import map file, css locations and polygon files."""
    model_data: ModelData = ImporterFactory.create(source, input_files.map_file).import_data()

    importer = ImportInputFiles()
    importer.set_logger(logger)
    cssdata = importer.css_file(input_files.css_file)

    regions = (
        RegionPolygon(input_files.region_file, logger=logger, default_value=default_region)
        if input_files.region_file else None
    )
    sections = (
        SectionPolygon(input_files.section_file, logger=logger, default_value=default_section)
        if input_files.section_file else None
    )

    return model_data, cssdata, regions, sections

def _classify_cross_sections_without_regions(model_data:ModelData, cssdata: dict) -> ModelData:
    """Classify without regions — called when no region file is present."""
    neigh = nearest_neighbour.get_class_tree(cssdata["xy"], cssdata["id"])
    model_data.geometry.sclass = neigh.predict(
        np.array([model_data.geometry.x, model_data.geometry.y]).T,
    )
    model_data.edges.sclass = neigh.predict(
        np.array([model_data.edges.x, model_data.edges.y]).T,
    )

    return model_data

def _classify_cross_sections_using_regions(model_data: ModelData, cssdata: dict, regions: RegionPolygon, input_files:InputFiles) -> ModelData:
    """Classify using region polygons — called when a region file is present."""
    gridpoints_in_regions: GridPointsInPolygonResults = regions.get_gridpoints_in_polygon(input_files.map_file)
    model_data.geometry.region = gridpoints_in_regions.faces_in_polygon
    model_data.edges.region = gridpoints_in_regions.edges_in_polygon

    css_regions = regions.get_points_in_polygon(cssdata["xy"], property_name="region")

    model_data.geometry.sclass = model_data.geometry.region.copy()
    model_data.edges.sclass = model_data.edges.region.copy()
    for region in np.unique(model_data.geometry.region):
        css_xy = cssdata["xy"][np.array(css_regions) == region]
        css_id = cssdata["id"][np.array(css_regions) == region]
        if len(css_id) == 0:
            continue
        neigh = nearest_neighbour.get_class_tree(css_xy, css_id)
        node_mask = model_data.geometry.region == region
        model_data.geometry.sclass[node_mask] = neigh.predict(
            np.array([model_data.geometry.x[node_mask], model_data.geometry.y[node_mask]]).T,
        )
        edge_mask = model_data.edges.region == region
        model_data.edges.sclass[edge_mask] = neigh.predict(
            np.array([model_data.edges.x[edge_mask], model_data.edges.y[edge_mask]]).T,
        )

    return model_data


def _classify_to_sections(model_data: ModelData, sections: SectionPolygon, map_file: Path) -> ModelData:
    """Classify 2D faces and edges to roughness sections."""
    if sections is None:
        model_data.edges.section    = classify_sections_by_variance(
            model_data.edges.section, model_data.hydraulics.chezy_edge,
        )
        model_data.geometry.section = classify_sections_by_variance(
            model_data.geometry.section, model_data.hydraulics.waterlevel,
        )
    else:
        gridpoints: GridPointsInPolygonResults = sections.get_gridpoints_in_polygon(map_file)
        model_data.geometry.section = gridpoints.faces_in_polygon
        model_data.edges.section    = gridpoints.edges_in_polygon

    return model_data



def classify_sections_by_variance(
    section: np.ndarray,
    variable: np.ndarray,
) -> np.ndarray:
    """Classify faces or edges into roughness sections using variance reduction on Chézy values.

    Assigns elements of ``section`` to ``"1"`` (main channel) or ``"2"`` (floodplain)
    by finding the variable split value that minimises within-group variance at
    the last timestep.

    Used when no section polygon file is provided.

    .. note::
        Variance reduction is a standard decision-tree splitting criterion.
        See https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction.

    Args:
        section:  1-D array of section labels to be updated (returned as new array).
        variable: 2-D array of values, shape ``(N_points, N_timesteps)``.

    Returns:
        Updated section array with values ``"1"`` (main) or ``"2"`` (floodplain).

    """
    end_values = variable[:, -1]  # last timestep, shape (N_points,)
    result = section.copy()
    threshold_no_variance = 2  # skip split when all values are near-identical

    split_candidates = np.arange(min(end_values), max(end_values), 1)
    if len(split_candidates) < threshold_no_variance:
        result[:] = "1"
    else:
        variance_list = [
            np.max(
                [
                    np.var(end_values[end_values > split]),
                    np.var(end_values[end_values <= split]),
                ],
            )
            for split in split_candidates
        ]
        splitpoint = split_candidates[np.nanargmin(variance_list)]
        result[end_values > splitpoint] = "1"   # main channel
        result[end_values <= splitpoint] = "2"  # floodplain

    return result
