"""Tests for export.py module."""

import pickle
from pathlib import Path

import pytest

from fm2prof.cross_section import CrossSection
from fm2prof.export import Export1DModelData
from tests.TestUtils import TestUtils

css_test_dir = "cross_sections"


class TestExport1DModelData:
    """Test class for Export1DModelData."""

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

        # Create and build cross-section
        css = CrossSection(data=css_data)
        css.build_geometry()
        css.calculate_correction()

        return css

    @pytest.fixture
    def exporter(self) -> Export1DModelData:
        """Create an Export1DModelData instance."""
        return Export1DModelData()

    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return output_dir

    def test_export_geometry_fm1d_creates_file(
        self,
        cross_section: CrossSection,
        exporter: Export1DModelData,
        temp_output_dir: Path,
    ) -> None:
        """Test that export_geometry creates output file for fm1d format."""
        # Arrange
        output_file = temp_output_dir / "geometry_fm1d.ini"
        cross_sections = [cross_section]

        # Act
        exporter.export_geometry(cross_sections, output_file, fmt="dflow1d")

        # Assert
        assert output_file.exists(), "Output file should be created"
        assert output_file.stat().st_size > 0, "Output file should not be empty"

    def test_export_roughness_fm1d_creates_file(
        self,
        cross_section: CrossSection,
        exporter: Export1DModelData,
        temp_output_dir: Path,
    ) -> None:
        """Test that export_roughness creates output file for fm1d format."""
        # Arrange
        output_file = temp_output_dir / "roughness_fm1d.ini"
        cross_sections = [cross_section]

        # Act
        exporter.export_roughness(
            cross_sections,
            output_file,
            fmt="dflow1d",
            roughness_section="Main",
        )

        # Assert
        assert output_file.exists(), "Output file should be created"
        assert output_file.stat().st_size > 0, "Output file should not be empty"

    def test_export_cross_section_locations_creates_file(
        self,
        cross_section: CrossSection,
        exporter: Export1DModelData,
        temp_output_dir: Path,
    ) -> None:
        """Test that export_cross_section_locations creates output file."""
        # Arrange
        output_file = temp_output_dir / "CrossSectionLocations.ini"
        cross_sections = [cross_section]

        # Act
        exporter.export_cross_section_locations(cross_sections, output_file)

        # Assert
        assert output_file.exists(), "Output file should be created"
        assert output_file.stat().st_size > 0, "Output file should not be empty"

    def test_export_volumes_creates_file(
        self,
        cross_section: CrossSection,
        exporter: Export1DModelData,
        temp_output_dir: Path,
    ) -> None:
        """Test that export_volumes creates output file."""
        # Arrange
        output_file = temp_output_dir / "volumes.csv"
        cross_sections = [cross_section]

        # Act
        exporter.export_volumes(cross_sections, output_file)

        # Assert
        assert output_file.exists(), "Output file should be created"
        assert output_file.stat().st_size > 0, "Output file should not be empty"

    def test_export_geometry_fm1d_file_content(
        self,
        cross_section: CrossSection,
        exporter: Export1DModelData,
        temp_output_dir: Path,
    ) -> None:
        """Test that exported fm1d geometry file contains expected content."""
        # Arrange
        output_file = temp_output_dir / "geometry_fm1d.ini"
        cross_sections = [cross_section]

        # Act
        exporter.export_geometry(cross_sections, output_file, fmt="dflow1d")

        # Assert
        content = output_file.read_text()
        assert "[General]" in content, "File should contain [General] section"
        assert "[Definition]" in content, "File should contain [Definition] section"
        assert "fileType" in content, "File should contain fileType parameter"
        assert cross_section.name in content, "File should contain cross-section name"

    def test_export_roughness_fm1d_file_content(
        self,
        cross_section: CrossSection,
        exporter: Export1DModelData,
        temp_output_dir: Path,
    ) -> None:
        """Test that exported fm1d roughness file contains expected content."""
        # Arrange
        output_file = temp_output_dir / "roughness_fm1d.ini"
        cross_sections = [cross_section]

        # Act
        exporter.export_roughness(
            cross_sections,
            output_file,
            fmt="dflow1d",
            roughness_section="Main",
        )

        # Assert
        content = output_file.read_text()
        assert "[General]" in content, "File should contain [General] section"
        assert "[Content]" in content, "File should contain [Content] section"
        assert "sectionId" in content, "File should contain sectionId parameter"
        assert "Main" in content, "File should reference Main section"
