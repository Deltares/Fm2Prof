"""GeoTIFF elevation importer for face geometry data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tifffile import TiffFile, TiffPage

from fm2prof.imports.base import BaseImporter, FaceGeometry, ModelData

if TYPE_CHECKING:
    from collections.abc import Iterator

TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_GDAL_NODATA = 42113
RASTER_DIMENSIONS = 2


def _blocks(page: TiffPage, nodata: float | None) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield cropped TIFF strips or tiles in raster order."""
    n_rows, n_cols = page.shape
    for segment, position, shape in page.segments(maxworkers=1):
        row, col = position[2:4]
        height = min(shape[1], n_rows - row)
        width = min(shape[2], n_cols - col)
        if segment is None:
            data = np.full((height, width), 0 if nodata is None else nodata, dtype=page.dtype)
        else:
            data = segment[0, :height, :width, 0]
        yield row, col, data


def _valid_cells(data: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = ~np.isnan(data)
    if nodata is not None and not np.isnan(nodata):
        valid &= data != nodata
    return valid


class GeoTifImporter(BaseImporter):
    """Importer for GeoTIFF elevation rasters."""

    SOURCE = "geotif"

    def import_data(self) -> ModelData:
        """Import geometry from a single-band GeoTIFF.

        Returns:
            Model data containing face geometry and no edges or hydraulics.

        Raises:
            ValueError: If the raster is not two-dimensional or lacks georeferencing tags.
        """
        self.set_logger_message(f"Reading geotif file: {self.file_path}")

        with TiffFile(self.file_path) as tif:
            page = tif.pages[0]
            if len(page.shape) != RASTER_DIMENSIONS or tif.series[0].shape != page.shape:
                msg = "GeoTIFF must contain a two-dimensional, single-band raster."
                raise ValueError(msg)

            nodata = self._get_nodata_value(tif)
            if TAG_MODEL_PIXEL_SCALE not in page.tags or TAG_MODEL_TIEPOINT not in page.tags:
                msg = "GeoTIFF is missing required georeferencing tags (PixelScale or TiePoints)."
                raise ValueError(msg)

            pixel_scale = page.tags[TAG_MODEL_PIXEL_SCALE].value
            tie_points = page.tags[TAG_MODEL_TIEPOINT].value
            x_origin, y_origin = tie_points[3:5]
            x_spacing, y_spacing = pixel_scale[0], -pixel_scale[1]

            # Count per row first so tiles can be written directly in row-major order.
            row_counts = np.zeros(page.shape[0], dtype=np.int64)
            for row, _, data in _blocks(page, nodata):
                row_counts[row : row + data.shape[0]] += np.count_nonzero(_valid_cells(data, nodata), axis=1)

            n_faces = int(row_counts.sum())
            row_offsets = np.cumsum(row_counts) - row_counts
            x = np.empty(n_faces, dtype=float)
            y = np.empty(n_faces, dtype=float)
            bedlevel = np.empty(n_faces, dtype=float)

            for row, col, data in _blocks(page, nodata):
                rows, cols = np.nonzero(_valid_cells(data, nodata))
                counts = np.bincount(rows, minlength=data.shape[0])
                block_offsets = np.cumsum(counts) - counts
                indices = row_offsets[row + rows] + np.arange(len(rows)) - block_offsets[rows]
                x[indices] = x_origin + (col + cols) * x_spacing
                y[indices] = y_origin + (row + rows) * y_spacing
                bedlevel[indices] = data[rows, cols]
                row_offsets[row : row + data.shape[0]] += counts

        geometry = FaceGeometry(
            x=x,
            y=y,
            bedlevel=bedlevel,
            area=np.full(n_faces, x_spacing * -y_spacing),
            section=np.full(n_faces, "main", dtype=object),
            region=np.full(n_faces, "", dtype=object),
            sclass=np.full(n_faces, "", dtype=object),
            islake=np.zeros(n_faces, dtype=bool),
        )
        return ModelData(
            geometry=geometry,
            cross_sections=[],
            source=self.SOURCE,
            edges=None,
            hydraulics=None,
        )

    def _get_nodata_value(self, tif: TiffFile) -> float | None:
        if TAG_GDAL_NODATA in tif.pages[0].tags:
            return float(tif.pages[0].tags[TAG_GDAL_NODATA].value)
        return None
