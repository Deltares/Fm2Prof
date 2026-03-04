"""Base exporter class for all format-specific exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from fm2prof.common import FM2ProfBase

if TYPE_CHECKING:
    from fm2prof.cross_section import CrossSection
    from fm2prof.export.output_files import OutputFileConfig


class BaseExporter(FM2ProfBase, ABC):
    """Abstract base class for all cross-section exporters.

    This class defines the common interface that all format-specific
    exporters must implement. It handles common functionality like
    volume export and file path management.

    Attributes:
        output_dir: Directory where output files will be written
        output_files: Configuration object for output file names
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        output_files: OutputFileConfig | None = None,
    ) -> None:
        """Initialise the exporter.

        Args:
            output_dir: Directory for output files. Defaults to current directory.
            output_files: Configuration for output file names. Uses format defaults if None.
        """
        super().__init__()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_files = output_files or self._default_output_files()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _default_output_files(self) -> OutputFileConfig:
        """Return default output file configuration for this format.

        Returns:
            OutputFileConfig instance with default file names
        """
        ...

    @abstractmethod
    def export_geometry(self, cross_sections: list[CrossSection]) -> Path:
        """Export cross-section geometry data.

        Args:
            cross_sections: List of cross-section objects to export

        Returns:
            Path to the created geometry file
        """
        ...

    @abstractmethod
    def export_roughness(self, cross_sections: list[CrossSection]) -> list[Path]:
        """Export roughness data.

        Args:
            cross_sections: List of cross-section objects to export

        Returns:
            List of paths to created roughness files
        """
        ...

    def export_volumes(self, cross_sections: list[CrossSection]) -> Path:
        """Export volume/water level comparison data.

        This method is common to all formats and exports a CSV file
        comparing 2D model volumes with 1D cross-section volumes.

        Args:
            cross_sections: List of cross-section objects to export

        Returns:
            Path to the created volumes file
        """
        file_path = self.output_files.get_file_path(self.output_dir, "volumes")

        with file_path.open("w") as f:
            # Write header
            f.write(
                "id,z,2D_total_volume,2D_flow_volume,2D_wet_area,2D_flow_area,"
                "1D_total_volume_sd,1D_total_volume,1D_flow_volume_sd,1D_flow_volume,"
                "1D_total_width,1D_flow_width\n",
            )

            for css in cross_sections:
                for i in range(len(css._css_z)):
                    f.write(
                        f"{css.name},{css._css_z[i]},{css._fm_total_volume[i]},"
                        f"{css._fm_flow_volume[i]},{css._fm_wet_area[i]},"
                        f"{css._fm_flow_area[i]},{css._css_total_volume_corrected[i]},"
                        f"{css._css_total_volume[i]},{css._css_flow_volume_corrected[i]},"
                        f"{css._css_flow_volume[i]},{css._css_total_width[i]},"
                        f"{css._css_flow_width[i]}\n",
                    )

        self.set_logger_message(f"Exported volumes to {file_path}", "info")
        return file_path

    def export_all(self, cross_sections: list[CrossSection]) -> dict[str, Path | list[Path]]:
        """Export all data types for the given cross-sections.

        Args:
            cross_sections: List of cross-section objects to export

        Returns:
            Dictionary mapping output type to file path(s)
        """
        results = {}
        results["geometry"] = self.export_geometry(cross_sections)
        results["roughness"] = self.export_roughness(cross_sections)
        results["volumes"] = self.export_volumes(cross_sections)
        return results
