"""Importer factory for creating format-specific importers."""

from __future__ import annotations

from pathlib import Path

from fm2prof.imports.base import BaseImporter


class ImporterFactory:
    """Factory for creating format-specific importers.

    Example:
        >>> importer = ImporterFactory.create("dflowfm", "path/to/map.nc")
        >>> data = importer.import_data()

    """

    _supported_sources = ("dflowfm",)

    @staticmethod
    def create(source: str, file_path: Path | str) -> BaseImporter:
        """Create an importer for the given source format.

        Args:
            source: Source format identifier (case-insensitive). Supported: 'dflowfm'.
            file_path: Path to the input file.

        Returns:
            A format-specific importer instance.

        Raises:
            NotImplementedError: If the source format is not supported.

        """
        from fm2prof.imports.dflowfm import DFlowFMImporter  # noqa: PLC0415 - avoid circular imports

        source_lower = source.lower()

        importers = {
            "dflowfm": DFlowFMImporter,
        }

        if source_lower not in importers:
            supported = ", ".join(f"'{s}'" for s in importers)
            msg = f"Unknown import source '{source}'. Supported sources are: {supported}"
            raise NotImplementedError(msg)

        return importers[source_lower](file_path)
