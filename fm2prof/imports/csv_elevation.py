"""CSV elevation importer for face geometry data."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from fm2prof.imports.base import BaseImporter, FaceGeometry, ModelData


class CsvElevationImporter(BaseImporter):
    """Importer for CSV files containing face geometry (elevation) data.

    Expected columns:
        - ``POINT_X``:   face centroid x-coordinate [m]
        - ``POINT_Y``:   face centroid y-coordinate [m]
        - ``POINT_Z``:   bed level [m+NAP]
        - ``Shape_Area``: face area [m²]

    This source only contains geometry — no edge or hydraulic data is available.
    ``ModelData.edges`` and ``ModelData.hydraulics`` will be ``None``.
    """

    SOURCE = "csv_elevation"

    COLUMN_X        = "POINT_X"
    COLUMN_Y        = "POINT_Y"
    COLUMN_BEDLEVEL = "POINT_Z"
    COLUMN_AREA     = "Shape_Area"

    REQUIRED_COLUMNS: ClassVar[set[str]] = {COLUMN_X, COLUMN_Y, COLUMN_BEDLEVEL, COLUMN_AREA}

    def import_data(self) -> ModelData:
        """Import face geometry from a CSV elevation file.

        Returns:
            :class:`~fm2prof.imports.base.ModelData` with ``geometry`` populated.
            ``edges`` and ``hydraulics`` are ``None``.

        Raises:
            ValueError: If any required columns are missing from the CSV.
        """
        self.set_logger_message(f"Reading CSV elevation file: {self.file_path}")

        faces = pd.read_csv(self.file_path)

        missing = self.REQUIRED_COLUMNS - set(faces.columns)
        if missing:
            msg = (
                f"CSV file is missing required columns: {sorted(missing)}. "
                f"Available columns: {list(faces.columns)}"
            )
            raise ValueError(msg)

        n_faces = len(faces)
        geometry = FaceGeometry(
            x=        faces[self.COLUMN_X].to_numpy(dtype=float),
            y=        faces[self.COLUMN_Y].to_numpy(dtype=float),
            bedlevel= faces[self.COLUMN_BEDLEVEL].to_numpy(dtype=float),
            area=     faces[self.COLUMN_AREA].fillna(0.0).to_numpy(dtype=float),
            # classification fields — populated later by the preprocessing pipeline
            section=  np.array(["main"]      * n_faces, dtype=object),
            region=   np.array([""]          * n_faces, dtype=object),
            sclass=   np.array([""]          * n_faces, dtype=object),
            islake=   np.zeros(n_faces,        dtype=bool),
        )

        self.set_logger_message(f"Imported {n_faces} faces from CSV elevation file")

        return ModelData(
            geometry=geometry,
            cross_sections=[],
            source=self.SOURCE,
            edges=None,
            hydraulics=None,
        )
