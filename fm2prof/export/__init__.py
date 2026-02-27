"""Export module for FM2PROF cross-section data.

This module provides exporters for converting FM2PROF cross-section data
into formats compatible with different hydraulic modelling software.

Supported Formats:
    - D-Flow 1D (FM1D): INI-based format for Deltares D-Flow 1D models
    - D-Hydro: Format for D-Hydro Suite models

Classes:
    BaseExporter: Abstract base class for all exporters
    DFlow1DExporter: Exporter for D-Flow 1D format
    DHydroExporter: Exporter for D-Hydro format
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
from fm2prof.export.dhydro import DHydroExporter
from fm2prof.export.factory import ExporterFactory
from fm2prof.export.output_files import DFlow1DOutputFiles, DHydroOutputFiles

__all__ = [
    "BaseExporter",
    "DFlow1DExporter",
    "DHydroExporter",
    "DFlow1DOutputFiles",
    "DHydroOutputFiles",
    "ExporterFactory",
]
