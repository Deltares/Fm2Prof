"""Tests for export module."""

import pickle
from pathlib import Path

import pytest

from fm2prof.cross_section import CrossSection
from fm2prof.export import DFlow1DExporter, ExporterFactory
from tests.TestUtils import TestUtils

css_test_dir = "cross_sections"


class TestExporters:
    """Test class for export module."""

    @pytest.fixture
    def cross_section(self) -> CrossSection:
        """Create a cross-section object for testing.

        Constructs a cross-section in the same way as test_reduce_points
        in test_crosssection.py.
        """
        # Load test data from pickle file
        test_case = {
            "name": "waal_1_40147.826",
        }
        tdir = TestUtils.get_local_test_data_dir(css_test_dir)
        pickle_file = tdir.joinpath(f"{test_case.get('name')}.pickle")
        with pickle_file.open("rb") as f:
            css_data = pickle.load(f)  # noqa: S301

        # Create and build cross-section (including geometry and roughness)
        css = CrossSection(data=css_data)
        css.build_geometry()
        css.calculate_correction()
        css.reduce_points(count_after=20)  # Must be called before assign_roughness
        css.assign_roughness()  # Add roughness tables

        return css

    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    def test_factory_create_dflow1d(self, temp_output_dir: Path) -> None:
        """Test that factory creates DFlow1DExporter."""
        exporter = ExporterFactory.create("dflow1d", output_dir=temp_output_dir)
        assert isinstance(exporter, DFlow1DExporter)

    def test_factory_unsupported_format(self, temp_output_dir: Path) -> None:
        """Test that factory raises error for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            ExporterFactory.create("unknown_format", output_dir=temp_output_dir)

    def test_dflow1d_export_geometry_creates_files(
        self,
        cross_section: CrossSection,
        temp_output_dir: Path,
    ) -> None:
        """Test that DFlow1D exporter creates geometry files."""
        exporter = DFlow1DExporter(output_dir=temp_output_dir)
        cross_sections = [cross_section]

        # Act
        result_path = exporter.export_geometry(cross_sections)

        # Assert
        assert result_path.exists(), "Geometry definitions file should be created"
        assert result_path.stat().st_size > 0, "File should not be empty"

        # Check that locations file was also created
        locations_file = temp_output_dir / "CrossSectionLocations.ini"
        assert locations_file.exists(), "Locations file should be created"

    def test_dflow1d_export_roughness_creates_files(
        self,
        cross_section: CrossSection,
        temp_output_dir: Path,
    ) -> None:
        """Test that DFlow1D exporter creates roughness files."""
        exporter = DFlow1DExporter(output_dir=temp_output_dir)
        cross_sections = [cross_section]

        # Act
        result_paths = exporter.export_roughness(cross_sections)

        # Assert
        assert len(result_paths) > 0, "At least one roughness file should be created"
        for path in result_paths:
            assert path.exists(), f"Roughness file {path} should exist"
            assert path.stat().st_size > 0, "File should not be empty"

    def test_dflow1d_export_volumes_creates_file(
        self,
        cross_section: CrossSection,
        temp_output_dir: Path,
    ) -> None:
        """Test that DFlow1D exporter creates volumes file."""
        exporter = DFlow1DExporter(output_dir=temp_output_dir)
        cross_sections = [cross_section]

        # Act
        result_path = exporter.export_volumes(cross_sections)

        # Assert
        assert result_path.exists(), "Volumes file should be created"
        assert result_path.stat().st_size > 0, "File should not be empty"

    def test_dflow1d_export_all(
        self,
        cross_section: CrossSection,
        temp_output_dir: Path,
    ) -> None:
        """Test that export_all creates all required files."""
        exporter = DFlow1DExporter(output_dir=temp_output_dir)
        cross_sections = [cross_section]

        # Act
        results = exporter.export_all(cross_sections)

        # Assert
        assert "geometry" in results, "Results should contain geometry key"
        assert "roughness" in results, "Results should contain roughness key"
        assert "volumes" in results, "Results should contain volumes key"

        # Check files exist
        assert results["geometry"].exists()
        assert all(p.exists() for p in results["roughness"])
        assert results["volumes"].exists()

    def test_dflow1d_geometry_file_content(
        self,
        cross_section: CrossSection,
        temp_output_dir: Path,
    ) -> None:
        """Test that exported DFlow1D geometry file contains expected content."""
        exporter = DFlow1DExporter(output_dir=temp_output_dir)
        cross_sections = [cross_section]

        # Act
        result_path = exporter.export_geometry(cross_sections)

        # Assert
        content = result_path.read_text()
        assert "[General]" in content, "File should contain [General] section"
        assert "[Definition]" in content, "File should contain [Definition] section"
        assert "fileType" in content, "File should contain fileType parameter"
        assert cross_section.name in content, "File should contain cross-section name"
        assert "levels" in content, "File should contain levels"
        assert "flowWidths" in content, "File should contain flowWidths"
        assert "totalWidths" in content, "File should contain totalWidths"
