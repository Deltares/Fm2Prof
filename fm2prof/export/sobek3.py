"""SOBEK 3 format exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fm2prof.export.base import BaseExporter
from fm2prof.export.output_files import Sobek3OutputFiles

if TYPE_CHECKING:
    from io import TextIOWrapper
    from pathlib import Path

    from fm2prof.cross_section import CrossSection
    from fm2prof.export.output_files import OutputFileConfig


class Sobek3Exporter(BaseExporter):
    """Exporter for SOBEK 3 CSV format.

    SOBEK 3 uses CSV files with specific column structures for
    geometry and roughness data. This exporter generates files
    compatible with SOBEK 3 hydraulic models.
    """

    def _default_output_files(self) -> OutputFileConfig:
        """Return default SOBEK 3 output file configuration.

        Returns:
            Sobek3OutputFiles instance with default file names
        """
        return Sobek3OutputFiles()

    def export_geometry(self, cross_sections: list[CrossSection]) -> Path:
        """Export geometry in SOBEK 3 CSV format.

        Args:
            cross_sections: List of cross-sections to export

        Returns:
            Path to created geometry file
        """
        file_path = self.output_files.get_file_path(self.output_dir, "geometry")

        with file_path.open("w") as f:
            self._write_geometry_header(f)

            for css in cross_sections:
                try:
                    self._write_cross_section_meta(f, css)
                    self._write_cross_section_geometry(f, css)
                except (ValueError, AttributeError, KeyError) as e:
                    self.set_logger_message(
                        f"Error exporting cross-section {css.name}: {e}",
                        "error",
                    )

        self.set_logger_message(f"Exported SOBEK 3 geometry to {file_path}", "info")
        return file_path

    def export_roughness(self, cross_sections: list[CrossSection]) -> list[Path]:
        """Export roughness in SOBEK 3 CSV format.

        Args:
            cross_sections: List of cross-sections to export

        Returns:
            List containing path to roughness file
        """
        file_path = self.output_files.get_file_path(self.output_dir, "roughness")

        with file_path.open("w") as f:
            self._write_roughness_header(f)

            # Get all unique sections across all cross-sections (numeric IDs)
            sections = np.unique([
                s for css in cross_sections
                for s in css.friction_tables
            ])

            for section_id in sections:
                for css in cross_sections:
                    if section_id in css.friction_tables:
                        self._write_roughness_section(f, css, section_id)

        self.set_logger_message(f"Exported SOBEK 3 roughness to {file_path}", "info")
        return [file_path]

    def _write_geometry_header(self, f: TextIOWrapper) -> None:
        """Write SOBEK 3 geometry CSV header.

        Args:
            f: File handle to write to
        """
        f.write(
            "id,Name,Data_type,level,Total width,Flow width,Profile_type,branch,"
            "chainage,width main channel,width floodplain 1,width floodplain 2,"
            "width sediment transport,Use Summerdike,Crest level summerdike,"
            "Floodplain baselevel behind summerdike,Flow area behind summerdike,"
            "Total area behind summerdike,Use groundlayer,Ground layer depth\n",
        )

    def _write_cross_section_meta(self, f: TextIOWrapper, css: CrossSection) -> None:
        """Write metadata row for a cross-section.

        Args:
            f: File handle to write to
            css: Cross-section object
        """
        # Determine summer dike parameters
        use_summerdike = "0"
        crest_level = ""
        floodplain_base = ""
        total_area = ""

        if css.extra_total_volume > 0:
            use_summerdike = "1"
            crest_level = str(css.crest_level)
            total_area = str(css.extra_total_area)

            floodplain_base = (
                str(css.floodplain_base)
                if not np.isnan(css.floodplain_base)
                else str(css.crest_level)  # Virtual summer dike
            )

        f.write(
            f"{css.name},,meta,,,,"
            f"ZW,{css.branch},{css.chainage},"
            f"{css.section_widths.get('main')},"
            f"{css.section_widths.get('floodplain1')},"
            f"{css.section_widths.get('floodplain2')},,"
            f"{use_summerdike},{crest_level},{floodplain_base},"
            f"{total_area},{total_area},,,,,\n",
        )

    def _write_cross_section_geometry(self, f: TextIOWrapper, css: CrossSection) -> None:
        """Write geometry data rows for a cross-section.

        Args:
            f: File handle to write to
            css: Cross-section object
        """
        # Add small increment to avoid unique z-value errors in SOBEK
        increment = np.array(range(1, css.z.size + 1)) * 1e-5
        z_value = css.z + increment

        for index, width in enumerate(css.total_width):
            flow_width = css.flow_width[index]
            f.write(
                f"{css.name},,geom,{z_value[index]:.8f},"
                f"{width},{flow_width},,,,,,,,,,,,,\n",
            )

    def _write_roughness_header(self, f: TextIOWrapper) -> None:
        """Write SOBEK 3 roughness CSV header.

        Args:
            f: File handle to write to
        """
        f.write(
            "Name,Chainage,RoughnessType,SectionType,Dependance,Interpolation,"
            "Pos/neg,R_pos_constant,Q_pos,R_pos_f(Q),H_pos,R_pos__f(h),"
            "R_neg_constant,Q_neg,R_neg_f(Q),H_neg,R_neg_f(h)\n",
        )

    def _write_roughness_section(
        self,
        f: TextIOWrapper,
        css: CrossSection,
        section_id: int,
    ) -> None:
        """Write roughness data for a specific section.

        Args:
            f: File handle to write to
            css: Cross-section object
            section_id: Section ID (1=main, 2=floodplain1, 3=floodplain2)

        Raises:
            ValueError: If section ID is not recognised
        """
        # Map numeric section IDs to SOBEK 3 names
        section_map = {
            1: "Main",
            2: "FloodPlain1",
            3: "FloodPlain2",
        }

        if section_id not in section_map:
            err_msg = f"Unknown section ID: {section_id}"
            raise ValueError(err_msg)

        plain = section_map[section_id]
        table = css.friction_tables[section_id]

        for level, friction in zip(table.level, table.friction, strict=True):
            f.write(
                f"{css.branch},{css.chainage},Chezy,{plain},"
                f"Waterlevel,Linear,Same,,,,"
                f"{level},{friction},,,,\n",
            )
