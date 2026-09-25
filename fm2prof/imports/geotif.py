"""CSV elevation importer for face geometry data."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from tifffile import TiffFile, imread

from fm2prof.imports.base import BaseImporter, FaceGeometry, ModelData

# Geotif tag constants (from the GeoTIFF specification)
TAG_MODEL_PIXEL_SCALE  = 33550 # Pixel size in each raster space axis
TAG_MODEL_TIEPOINT     = 33922 # Maps raster (row, col) → model (x, y) space
TAG_GEO_DIRECTORY_KEY  = 34735 # index of "GeoKeys" (the actual CRS/projection metadata)
TAG_GEO_DOUBLE_PARAMS  = 34736 # Double-precision values referenced by GeoKeys
TAG_GEO_ASCII_PARAMS   = 34737 # String values referenced by GeoKeys
TAG_GDAL_NODATA        = 42113 # GDAL-specific (not part of official GeoTIFF spec) — encodes the NoData value as an ASCII string


class GeoTifImporter(BaseImporter):
    """Importer for geotif files"""

    SOURCE = "geotif"

    def import_data(self) -> ModelData:
        """Import data from a geotif file.

        Returns:
            :class:`~fm2prof.imports.base.ModelData` with ``geometry`` populated.
            ``edges`` and ``hydraulics`` are ``None``.

        Raises:
            ValueError: If any required columns are missing from the CSV.
        """
        self.set_logger_message(f"Reading geotif file: {self.file_path}")

        with TiffFile(self.file_path) as tif:

            # Get the NoData value if present
            nodata_value = self._get_nodata_value(tif)

            # Get pixel scale and tie points for georeferencing
            pixel_scale = None
            tie_points = None
            if TAG_MODEL_PIXEL_SCALE in tif.pages[0].tags:
                pixel_scale = tif.pages[0].tags[TAG_MODEL_PIXEL_SCALE].value
            if TAG_MODEL_TIEPOINT in tif.pages[0].tags:
                tie_points = tif.pages[0].tags[TAG_MODEL_TIEPOINT].value

            # construct x, y vectors
            if pixel_scale is None or tie_points is None:
                raise ValueError("GeoTIFF is missing required georeferencing tags (PixelScale or TiePoints).")

            x_origin = tie_points[3]
            y_origin = tie_points[4]
            x_spacing = pixel_scale[0]
            y_spacing = -pixel_scale[1] # tag stores positive magnitude; convert to signed north-up convention

            # Read the raster data
            raster_data = tif.asarray()

            n_rows, n_cols = raster_data.shape
            x_coords = x_origin + np.arange(n_cols) * x_spacing
            y_coords = y_origin + np.arange(n_rows) * y_spacing

            # Create a meshgrid of coordinates
            xv, yv = np.meshgrid(x_coords, y_coords)

            # Discard all no-data values using a mask for valid numbers
            valid_mask = raster_data != nodata_value
            n_faces = valid_mask.sum()

            geometry = FaceGeometry(
            x =        xv[valid_mask],
            y =        yv[valid_mask],
            bedlevel = raster_data[valid_mask],
            area =     x_spacing*-y_spacing*np.ones(n_faces),
            # classification fields — populated later by the preprocessing pipeline
            section =  np.array(["main"]      * n_faces, dtype=object),
            region =   np.array([""]          * n_faces, dtype=object),
            sclass =   np.array([""]          * n_faces, dtype=object),
            islake =   np.zeros(n_faces,        dtype=bool),
            )

            return ModelData(
                geometry=geometry,
                cross_sections=[],
                source=self.SOURCE,
                edges=None,
                hydraulics=None,
            )


    def _get_nodata_value(self, tif: TiffFile) -> float | None:
        nodata_value = None
        if TAG_GDAL_NODATA in tif.pages[0].tags:
            nodata_str = tif.pages[0].tags[TAG_GDAL_NODATA].value
            try:
                nodata_value = float(nodata_str)
            except ValueError:
                self.set_logger_message(
                    f"Could not convert GDAL NoData value '{nodata_str}' to float.",
                    "warning"
                )

        return nodata_value