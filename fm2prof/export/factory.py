"""Factory for creating format-specific exporters."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from fm2prof.export.dflow1d import DFlow1DExporter
from fm2prof.export.sobek3 import Sobek3Exporter

if TYPE_CHECKING:
    from pathlib import Path

    from fm2prof.export.base import BaseExporter
    from fm2prof.export.output_files import OutputFileConfig


class ExporterFactory:
    """Factory for creating appropriate exporter instances.

    This factory simplifies the creation of format-specific exporters
    and provides a central registry of supported formats.

    Example:
        >>> from fm2prof.export import ExporterFactory
        >>> exporter = ExporterFactory.create('sobek3', output_dir='./output')
        >>> exporter.export_all(cross_sections)
    """

    _exporters: ClassVar[dict] = {
        "sobek3": Sobek3Exporter,
        "dflow1d": DFlow1DExporter,
        "fm1d": DFlow1DExporter,  # Alias for dflow1d
    }

    @classmethod
    def create(
        cls,
        fmt: str,
        output_dir: Path | str | None = None,
        output_files: OutputFileConfig | None = None,
    ) -> BaseExporter:
        """Create an exporter for the specified format.

        Args:
            fmt: Format identifier ('sobek3', 'dflow1d', or 'fm1d')
            output_dir: Directory for output files
            output_files: Custom output file configuration

        Returns:
            Appropriate exporter instance

        Raises:
            ValueError: If format is not supported
        """
        fmt_lower = fmt.lower()

        if fmt_lower not in cls._exporters:
            supported = ", ".join(cls._exporters.keys())
            err_msg = f"Unsupported format '{fmt}'. Supported formats: {supported}"
            raise ValueError(err_msg)

        exporter_class = cls._exporters[fmt_lower]
        return exporter_class(output_dir=output_dir, output_files=output_files)

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Get list of supported export formats.

        Returns:
            List of format identifiers
        """
        return list(cls._exporters.keys())
