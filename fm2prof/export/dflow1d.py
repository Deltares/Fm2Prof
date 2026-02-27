"""D-Flow 1D format exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fm2prof.export.base import BaseExporter
from fm2prof.export.output_files import DFlow1DOutputFiles

if TYPE_CHECKING:
    from io import TextIOWrapper
    from pathlib import Path

    from fm2prof.cross_section import CrossSection
    from fm2prof.export.output_files import OutputFileConfig


class DFlow1DExporter(BaseExporter):
    """Exporter for D-Flow 1D INI format.

    D-Flow 1D uses INI-style configuration files with specific
    sections and parameters. This exporter generates files
    compatible with Deltares D-Flow 1D hydraulic models.
    """

    def _default_output_files(self) -> OutputFileConfig:
        """Return default D-Flow 1D output file configuration.

        Returns:
            DFlow1DOutputFiles instance with default file names
        """
        return DFlow1DOutputFiles()

    def export_geometry(self, cross_sections: list[CrossSection]) -> Path:
        """Export geometry in D-Flow 1D INI format.

        Args:
            cross_sections: List of cross-sections to export

        Returns:
            Path to created cross-section definitions file
        """
        # Export cross-section definitions
        def_path = self.output_files.get_file_path(self.output_dir, "css_definitions")

        with def_path.open("w") as f:
            self._write_geometry_header(f)

            for css in cross_sections:
                self._write_cross_section_definition(f, css)

        # Export cross-section locations
        loc_path = self.export_cross_section_locations(cross_sections)

        self.set_logger_message(
            f"Exported D-Flow 1D geometry to {def_path} and {loc_path}",
            "info",
        )
        return def_path

    def export_roughness(self, cross_sections: list[CrossSection]) -> list[Path]:
        """Export roughness in D-Flow 1D INI format.

        Exports separate files for main channel and floodplains.

        Args:
            cross_sections: List of cross-sections to export

        Returns:
            List of paths to created roughness files
        """
        roughness_paths = []

        # Map numeric section IDs to section names
        section_map = {
            "1": "main",
            "2": "floodplain1",
            "3": "floodplain2",
        }

        # Export roughness for each section type
        for section_id, section_name in section_map.items():
            # Check if any cross-section has this section
            has_section = any(
                section_id in css.friction_tables
                for css in cross_sections
            )

            if not has_section:
                continue

            file_path = self.output_files.get_file_path(
                self.output_dir,
                f"roughness_{section_name}",
            )

            with file_path.open("w") as f:
                self._write_roughness_file(f, cross_sections, section_id)

            roughness_paths.append(file_path)
            self.set_logger_message(
                f"Exported D-Flow 1D roughness ({section_name}) to {file_path}",
                "info",
            )

        return roughness_paths

    def export_cross_section_locations(
        self,
        cross_sections: list[CrossSection],
    ) -> Path:
        """Export cross-section locations in D-Flow 1D format.

        Args:
            cross_sections: List of cross-sections to export

        Returns:
            Path to created locations file
        """
        file_path = self.output_files.get_file_path(self.output_dir, "css_locations")

        with file_path.open("w") as f:
            # Write general section
            f.write(
                "[General]\n"
                "majorVersion\t\t\t= 1\n"
                "minorversion\t\t\t= 0\n"
                "fileType\t\t\t\t= crossLoc\n\n",
            )

            for css in cross_sections:
                f.write(
                    "[CrossSection]\n"
                    f"\tid\t\t\t\t\t= {css.name}\n"
                    f"\tbranchid\t\t\t= {css.branch}\n"
                    f"\tchainage\t\t\t= {css.chainage}\n"
                    "\tshift\t\t\t\t= 0.000\n"
                    f"\tdefinition\t\t\t= {css.name}\n\n",
                )

        return file_path

    def _write_geometry_header(self, f: TextIOWrapper) -> None:
        """Write D-Flow 1D geometry file header.

        Args:
            f: File handle to write to
        """
        f.write(
            "[General]\n"
            "majorVersion\t\t\t= 1\n"
            "minorversion\t\t\t= 0\n"
            "fileType\t\t\t\t= crossDef\n\n",
        )

    def _write_cross_section_definition(
        self,
        f: TextIOWrapper,
        css: CrossSection,
    ) -> None:
        """Write a single cross-section definition.

        Args:
            f: File handle to write to
            css: Cross-section object
        """
        z = [f"{iz:.4f}" for iz in css.z]
        fw = [f"{iz:.4f}" for iz in css.flow_width]
        tw = [f"{iz:.4f}" for iz in css.total_width]

        f.write(
            "[Definition]\n"
            f"\tid\t\t\t\t\t= {css.name}\n"
            "\ttype\t\t\t\t= tabulated\n"
            "\tthalweg\t\t\t\t= 0.000\n"
            f"\tnumLevels\t\t\t= {len(z)}\n"
            f"\tlevels\t\t\t\t= {' '.join(z)}\n"
            f"\tflowWidths\t\t\t= {' '.join(fw)}\n"
            f"\ttotalWidths\t\t\t= {' '.join(tw)}\n"
            f"\tsd_crest\t\t\t= {css.crest_level:.4f}\n"
            f"\tsd_flowArea\t\t\t= {css.extra_flow_area}\n"
            f"\tsd_totalArea\t\t= {css.extra_total_area:.4f}\n"
            f"\tsd_baseLevel\t\t= {css.floodplain_base}\n"
            f"\tmain\t\t\t\t= {css.section_widths['main']:.4f}\n"
            f"\tfloodPlain1\t\t\t= {css.section_widths['floodplain1']:.4f}\n"
            f"\tfloodPlain2\t\t\t= {css.section_widths['floodplain2']:.4f}\n"
            "\tgroundlayerUsed\t\t= 0\n"
            "\tgroundLayer\t\t\t= 0.000\n\n",
        )

    def _write_roughness_file(
        self,
        f: TextIOWrapper,
        cross_sections: list[CrossSection],
        section_id: int,
    ) -> None:
        """Write complete roughness file for a section.

        Args:
            f: File handle to write to
            cross_sections: List of cross-sections
            section_id: Section ID (1=main, 2=floodplain1, 3=floodplain2)
        """
        # Map section ID to name
        section_map = {
            1: "main",
            2: "floodplain1",
            3: "floodplain2",
        }
        section_name = section_map.get(section_id, "main").capitalize()

        # Write general section
        f.write(
            "[General]\n"
            "\tmajorVersion          = 1\n"
            "\tminorVersion          = 0\n"
            "\tfileType              = roughness\n\n"
            "[Content]\n"
            f"\tsectionId             = {section_name}\n"
            "\tflowDirection         = False\n"
            "\tinterpolate           = 1\n"
            "\tglobalType            = 1\n"
            "\tglobalValue           = 45\n\n",
        )

        # Write branch properties
        branch_list = []
        for css in cross_sections:
            if section_id in css.friction_tables and css.branch not in branch_list:
                branch_list.append(css.branch)
                table = css.friction_tables[section_id]

                f.write(
                    "[BranchProperties]\n"
                    f"\tbranchId              = {css.branch}\n"
                    "\troughnessType         = 1\n"
                    "\tfunctionType          = 2\n"
                    f"\tnumLevels             = {len(table.level)}\n"
                    f"\tlevels                = {' '.join(f'{level:.4f}' for level in table.level)}\n\n",
                )

        # Write definitions
        for css in cross_sections:
            if section_id in css.friction_tables:
                table = css.friction_tables[section_id]

                f.write(
                    "[Definition]\n"
                    f"\tbranchId              = {css.branch}\n"
                    f"\tchainage              = {css.chainage}\n"
                    f"\tvalues                = {' '.join(f'{v:.4}' for v in table.friction)}\n\n",
                )
