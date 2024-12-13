"""Module for handling mask output file.

Copyright (C) Stichting Deltares 2019. All rights reserved.

This file is part of the Fm2Prof.

The Fm2Prof is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import geojson


def create_mask_point(coords: geojson.coords, properties: dict | None = None) -> geojson.Feature:
    """Create a Point based on the properties and coordinates given.

    Args:
        coords (geojson.coords): Coordinates tuple (x,y) for the mask point.
        properties (dict): Dictionary of properties

    """
    if not coords:
        err_msg = "coords cannot be empty."
        raise ValueError(err_msg)
    output_mask = geojson.Feature(geometry=geojson.Point(coords))

    if properties:
        output_mask.properties = properties
    return output_mask


def validate_extension(file_path: str | Path) -> None:
    """Validate extension of mask output file.

    Args:
        file_path (Union[str, Path]): path to output file.

    """
    if not isinstance(file_path, (str, Path)):
        err_msg = f"file_path should be string or Path, not {type(file_path)}"
        raise TypeError(err_msg)
    if Path(file_path).suffix not in [".json", ".geojson"]:
        err_msg = "Invalid file path extension, should be .json or .geojson."
        raise OSError(err_msg)


def read_mask_output_file(file_path: str | Path) -> dict:
    """Import a GeoJson from a given json file path.

    Args:
        file_path (str | Path): Location of the json file

    """
    file_path = Path(file_path)
    if not file_path.exists():
        err_msg = f"File path {file_path} not found."
        raise FileNotFoundError(err_msg)

    validate_extension(file_path)
    with file_path.open("r") as geojson_file:
        geojson_data = geojson.load(geojson_file)
    if not isinstance(geojson_data, geojson.FeatureCollection):
        err_msg = "File is empty or not a valid geojson file."
        raise OSError(err_msg)
    return geojson_data


def write_mask_output_file(file_path: Path | str, mask_points: list) -> None:
    """Write a geojson file with a Feature collection containing the mask_points list given as input.

    Arguments:
        file_path (str): file_path where to store the geojson.
        mask_points (list): List of features to output.

    """
    if not file_path:
        err_msg = "file_path is required."
        raise ValueError(err_msg)
    file_path = Path(file_path)
    if not mask_points:
        err_msg = "mask_points cannot be empty."
        raise ValueError(err_msg)
    validate_extension(file_path)
    feature_collection = geojson.FeatureCollection(mask_points)
    with file_path.open("w") as f:
        geojson.dump(feature_collection, f, indent=4)
