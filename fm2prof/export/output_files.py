"""Output file configuration for different export formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputFileConfig(ABC):
    """Base configuration for output files."""

    @abstractmethod
    def get_file_path(self, output_dir: Path, file_type: str) -> Path:
        """Get full file path for a specific output type.

        Args:
            output_dir: Directory where files should be written
            file_type: Type of file (e.g., 'geometry', 'roughness')

        Returns:
            Full path to the output file
        """
        ...


@dataclass
class Sobek3OutputFiles(OutputFileConfig):
    """Output file configuration for SOBEK 3 format."""

    geometry: str = "geometry.csv"
    roughness: str = "roughness.csv"
    volumes: str = "volumes.csv"

    def get_file_path(self, output_dir: Path, file_type: str) -> Path:
        """Get full file path for SOBEK 3 output.

        Args:
            output_dir: Directory where files should be written
            file_type: Type of file ('geometry', 'roughness', or 'volumes')

        Returns:
            Full path to the output file

        Raises:
            KeyError: If file_type is not recognised
        """
        file_map = {
            "geometry": self.geometry,
            "roughness": self.roughness,
            "volumes": self.volumes,
        }
        if file_type not in file_map:
            err_msg = f"Unknown file type '{file_type}' for SOBEK 3 format"
            raise KeyError(err_msg)
        return output_dir / file_map[file_type]


@dataclass
class DFlow1DOutputFiles(OutputFileConfig):
    """Output file configuration for D-Flow 1D format."""

    css_locations: str = "CrossSectionLocations.ini"
    css_definitions: str = "CrossSectionDefinitions.ini"
    roughness_main: str = "roughness-Main.ini"
    roughness_floodplain1: str = "roughness-FloodPlain1.ini"
    roughness_floodplain2: str = "roughness-FloodPlain2.ini"
    volumes: str = "volumes.csv"

    def get_file_path(self, output_dir: Path, file_type: str) -> Path:
        """Get full file path for D-Flow 1D output.

        Args:
            output_dir: Directory where files should be written
            file_type: Type of file (e.g., 'css_locations', 'roughness_main')

        Returns:
            Full path to the output file

        Raises:
            KeyError: If file_type is not recognised
        """
        file_map = {
            "css_locations": self.css_locations,
            "css_definitions": self.css_definitions,
            "roughness_main": self.roughness_main,
            "roughness_floodplain1": self.roughness_floodplain1,
            "roughness_floodplain2": self.roughness_floodplain2,
            "volumes": self.volumes,
        }
        if file_type not in file_map:
            err_msg = f"Unknown file type '{file_type}' for D-Flow 1D format"
            raise KeyError(err_msg)
        return output_dir / file_map[file_type]
