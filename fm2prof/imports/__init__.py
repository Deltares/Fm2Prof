"""FM2PROF import module.

Provides importers for reading 2D model data from various formats.

Example:
    >>> from fm2prof.imports import ImporterFactory
    >>> importer = ImporterFactory.create("dflowfm", "path/to/map.nc")
    >>> data = importer.import_data()
    >>> selection = data.get_selection("css_001")

"""

from fm2prof.imports.base import BaseImporter, ModelData
from fm2prof.imports.factory import ImporterFactory, detect_source

__all__ = ["BaseImporter", "ImporterFactory", "ModelData", "detect_source"]
