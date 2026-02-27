"""Export module for FM2PROF cross-section data.

This module provides exporters for converting FM2PROF cross-section data
into formats compatible with different hydraulic modelling software.

Supported Formats:
    - SOBEK 3: CSV-based format for SOBEK 3 hydraulic models
    - D-Flow 1D (FM1D): INI-based format for Deltares D-Flow 1D models

Classes:
    BaseExporter: Abstract base class for all exporters
    Sobek3Exporter: Exporter for SOBEK 3 format
    DFlow1DExporter: Exporter for D-Flow 1D format
    ExporterFactory: Factory for creating appropriate exporter instances

Example:
    >>> from fm2prof.export import ExporterFactory
    >>>
    >>> # Create exporter for specific format
    >>> exporter = ExporterFactory.create('dflow1d', output_dir='./output')
    >>>
    >>> # Export cross-sections
    >>> exporter.export_all(cross_sections)
"""

from fm2prof.export.base import BaseExporter
from fm2prof.export.dflow1d import DFlow1DExporter
from fm2prof.export.factory import ExporterFactory
from fm2prof.export.output_files import DFlow1DOutputFiles, Sobek3OutputFiles
from fm2prof.export.sobek3 import Sobek3Exporter

__all__ = [
    "BaseExporter",
    "DFlow1DExporter",
    "DFlow1DOutputFiles",
    "ExporterFactory",
    "Sobek3Exporter",
    "Sobek3OutputFiles",
]
