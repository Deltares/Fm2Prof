"""Classes for exporting data."""

from __future__ import annotations

# import from standar library
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

# import from package
from fm2prof.common import FM2ProfBase

if TYPE_CHECKING:
    from io import TextIOWrapper

    from fm2prof.cross_section import CrossSection


@dataclass
class OutputFiles:
    """Class for grouping output files."""

    dimr_css_locations: str = "CrossSectionLocations.ini"
    dimr_css_definitions: str = "CrossSectionDefinitions.ini"
    dimr_roughness_main: str = "roughness-Main.ini"
    dimr_roughness_floodplain1: str = "roughness-FloodPlain1.ini"
    dimr_roughness_floodplain2: str = "roughness-FloodPlain2.ini"
    sobek3_geometry: str = "geometry.csv"
    sobek3_roughness: str = "roughness.csv"
    test_geometry: str = "geometry_test.csv"
    test_roughness: str = "roughness_test.csv"
    fm2prof_volume: str = "volumes.csv"

    def __getitem__(self, index: int) -> str:
        """Get item by index from OutputFiles dataclass."""
        return astuple(self)[index]

    def __iter__(self) -> Iterator:
        """Get iterator of OutputFiles items."""
        return iter(astuple(self))


class Export1DModelData(FM2ProfBase):
    """Contains all functions related to exporting to various outputs.

    In the future, split in different classes for SOBEK, D-Hydro etc formats
    """

    def export_geometry(
        self,
        cross_sections: list[CrossSection],
        file_path: Path | str,
        fmt: str = "sobek3",
    ) -> None:
        """Export cross section geometries in different formats.

        Args:
        ----
            cross_sections (list[CrossSection]): List of cross sections.
            file_path (Path | str): File path to write to.
            fmt (str): Format to write to. Options are sobek3, dflow1d, and testformat.
                        Defaults to sobek3

        """
        with Path(file_path).open("w") as f:
            if fmt == "sobek3":
                """SOBEK 3 style csv"""
                self._write_geometry_sobek3(f, cross_sections)
            elif fmt == "dflow1d":
                """DFM 1D style"""
                self._write_geometry_fm1d(f, cross_sections)
            elif fmt == "testformat":
                """test format for system tests, only has geometry (no summerdike)"""
                self._write_geometry_testformat(f, cross_sections)

    def export_roughness(
        self,
        cross_sections: list[CrossSection],
        file_path: str | Path,
        fmt: str = "sobek3",
        roughness_section: str = "Main",
    ) -> None:
        """Export roughnes in different formats.

        Args:
        ----
            cross_sections (list[CrossSection]): List of cross sections
            file_path (str | Path): File path to write to
            fmt (str, optional): Format to write to. Defaults to "sobek3".
            roughness_section (str, optional): Name of roughness section. Defaults to "Main".

        """
        with Path(file_path).open("w") as f:
            if fmt == "sobek3":
                """SOBEK 3 style csv"""
                self._write_roughness_sobek3(f, cross_sections)
            elif fmt == "dflow1d":
                """DFM 1D style"""
                self._write_roughness_fm1d(f, cross_sections, roughness_section)
            elif fmt == "testformat":
                """test format for system tests, only has geometry (no summerdike)"""
                self._write_roughness_testformat(f, cross_sections)

    def export_volumes(
        self,
        cross_sections: list[CrossSection],
        file_path: str | Path,
    ) -> None:
        """Write to file the volume/waterlevel information.

        Args:
        ----
            cross_sections (list[CrossSection]): List of cross sections.
            file_path (str | Path): File path to write to.

        """
        with Path(file_path).open("w") as f:
            # Write header
            f.write(
                "id,z,2D_total_volume,2D_flow_volume,2D_wet_area,2D_flow_area,1D_total_volume_sd,1D_total_volume,1D_flow_volume_sd,1D_flow_volume,1D_total_width,1D_flow_width\n",
            )

            for css in cross_sections:
                for i in range(len(css._css_z)):
                    outputdata = {
                        "id": css.name,
                        "z": css._css_z[i],
                        "tv2d": css._fm_total_volume[i],
                        "fv2d": css._fm_flow_volume[i],
                        "wa2d": css._fm_wet_area[i],
                        "fa2d": css._fm_flow_area[i],
                        "tvsd1d": css._css_total_volume_corrected[i],
                        "tv1d": css._css_total_volume[i],
                        "fvsd1d": css._css_flow_volume_corrected[i],
                        "fv1d": css._css_flow_volume[i],
                        "tw1d": css._css_total_width[i],
                        "fw1d": css._css_flow_width[i],
                    }

                    f.write(
                        "{id},{z},{tv2d},{fv2d},{wa2d},{fa2d},{tvsd1d},{tv1d},{fvsd1d},{fv1d},{tw1d},{fw1d}\n".format(
                            **outputdata,
                        ),
                    )

    def export_cross_section_locations(
        self,
        cross_sections: list[CrossSection],
        file_path: str | Path,
    ) -> None:
        """Export cross section locations.

        Args:
        ----
            cross_sections (list[CrossSection]): List of cross sections.
            file_path (str | Path): file path to write to.

        """
        with Path(file_path).open("w") as fid:
            # Write general secton
            fid.write(
                "[General]\nmajorVersion\t\t\t= 1\nminorversion\t\t\t= 0\nfileType\t\t\t\t= crossLoc\n\n",
            )

            for css in cross_sections:
                fid.write("[CrossSection]\n")
                fid.write(f"\tid\t\t\t\t\t= {css.name}\n")
                fid.write(f"\tbranchid\t\t\t= {css.branch}\n")
                fid.write(f"\tchainage\t\t\t= {css.chainage}\n")
                fid.write("\tshift\t\t\t\t= 0.000\n")
                fid.write(f"\tdefinition\t\t\t= {css.name}\n\n")

    """ test file formats """

    def _write_geometry_testformat(
        self,
        fid: TextIOWrapper,
        cross_sections: list[CrossSection],
    ) -> None:
        # write header
        fid.write("chainage,z,total width,flow width\n")
        for _index, cross_section in enumerate(cross_sections):
            for i in range(len(cross_section.z)):
                fid.write(
                    f"{cross_section.chainage}, {cross_section.z[i]}, {cross_section.total_width[i]},"
                    f"{cross_section.flow_width[i]}\n",
                )

    def _write_roughness_testformat(
        self,
        fid: TextIOWrapper,
        cross_sections: list[CrossSection],
    ) -> None:
        # write header
        fid.write("chainage,type,waterlevel,chezy roughness\n")

        for roughnesstype in ("alluvial", "nonalluvial"):
            for cross_section in cross_sections:
                waterlevels = cross_section.alluvial_friction_table[0]

                if roughnesstype == "alluvial":
                    table = cross_section.alluvial_friction_table
                elif roughnesstype == "nonalluvial":
                    table = cross_section.nonalluvial_friction_table

                for index, level in enumerate(waterlevels):
                    try:
                        chezy = table[1][index]
                    except IndexError:
                        break
                    if not np.isnan(chezy):
                        fid.write(
                            f"{cross_section.chainage}, {roughnesstype}, {level}, {chezy}\n",
                        )

    """ FM 1D file formats """

    def _write_geometry_fm1d(
        self,
        fid: TextIOWrapper,
        cross_sections: list[CrossSection],
    ) -> None:
        """FM1D uses a configuration 'Delft' file style format."""
        # Write general secton
        fid.write(
            "[General]\nmajorVersion\t\t\t= 1\nminorversion\t\t\t= 0\nfileType\t\t\t\t= crossDef\n\n",
        )

        for css in cross_sections:
            z = [f"{iz:.4f}" for iz in css.z]
            fw = [f"{iz:.4f}" for iz in css.flow_width]
            tw = [f"{iz:.4f}" for iz in css.total_width]

            fid.write("[Definition]\n")
            fid.write(
                f"\tid\t\t\t\t\t= {css.name}\n"  # noqa: ISC003
                "\ttype\t\t\t\t= tabulated\n"
                "\tthalweg\t\t\t\t= 0.000\n"
                + f"\tnumLevels\t\t\t= {len(z)}\n"
                + "\tlevels\t\t\t\t= {}\n".format(" ".join(z))
                + "\tflowWidths\t\t\t= {}\n".format(" ".join(fw))
                + "\ttotalWidths\t\t\t= {}\n".format(" ".join(tw))
                + f"\tsd_crest\t\t\t= {css.crest_level:.4f}\n"
                + f"\tsd_flowArea\t\t\t= {css.extra_flow_area}\n"
                + f"\tsd_totalArea\t\t= {css.extra_total_area:.4f}\n"
                + f"\tsd_baseLevel\t\t= {css.floodplain_base}\n"
                + "\tmain\t\t\t\t= {:.4f}\n".format(css.section_widths["main"])
                + "\tfloodPlain1\t\t\t= {:.4f}\n".format(
                    css.section_widths["floodplain1"],
                )
                + "\tfloodPlain2\t\t\t= {:.4f}\n".format(
                    css.section_widths["floodplain2"],
                )
                + "\tgroundlayerUsed\t\t= 0\n"
                + "\tgroundLayer\t\t\t= 0.000\n\n",
            )

    def _write_roughness_fm1d(
        self,
        fid: TextIOWrapper,
        cross_sections: list[CrossSection],
        section: str,
    ) -> None:
        general_sec = f"""[General]
        majorVersion          = 1
        minorVersion          = 0
        fileType              = roughness

    [Content]
        sectionId             = {section}
        flowDirection         = False
        interpolate           = 1
        globalType            = 1
        globalValue           = 45

    """

        branch_sec = self._get_fm1d_branch_sec(cross_sections, section.lower())
        definition_sec = self._get_fm1d_definition_sec(cross_sections, section.lower())
        fid.write(general_sec)
        fid.write(branch_sec)
        fid.write(definition_sec)

    def _get_fm1d_definition_sec(
        self,
        cross_sections: list[CrossSection],
        section: str,
    ) -> str:
        def_sec = ""

        for css in cross_sections:
            if section in css.friction_tables:
                table = css.friction_tables[section]
                def_sec += """[Definition]
        branchId              = {}
        chainage              = {}
        values                = {}

    """.format(
                    css.branch,
                    css.chainage,
                    " ".join(map("{:.4}".format, table.friction)),
                )
        return def_sec

    def _get_fm1d_branch_sec(
        self,
        cross_sections: list[CrossSection],
        section: str,
    ) -> str:
        branch_sec = ""
        branch_list = []
        for css in cross_sections:
            if section in css.friction_tables and css.branch not in branch_list:
                branch_list.append(css.branch)
                branch_sec += """[BranchProperties]
        branchId              = {}
        roughnessType         = 1
        functionType          = 2
        numLevels             = {}
        levels                = {}
""".format(
                    css.branch,
                    len(css.friction_tables[section].level),
                    " ".join(
                        map("{:.4f}".format, css.friction_tables[section].level),
                    ),
                )

        return branch_sec

    """ SOBEK 3 file formats """

    def _write_geometry_sobek3(
        self,
        fid: TextIOWrapper,
        cross_sections: list[CrossSection],
    ) -> None:
        # write meta
        # note, the chainage is currently set to the X-coordinate of the cross-section (straight channel)
        # note, the channel naming strategy must be discussed, currently set to 'Channel' for all cross-sections

        # write header
        fid.write(
            "id,Name,Data_type,level,Total width,Flow width,Profile_type,branch,chainage,width main channel,width"
            "floodplain 1,width floodplain 2,width sediment transport,Use Summerdike,Crest level summerdike,Floodplain"
            "baselevel behind summerdike,Flow area behind summerdike,Total area behind summerdike,Use groundlayer,"
            "Ground layer depth\n",
        )

        for cross_section in cross_sections:
            try:
                b_summerdike = "0"
                crest_level = ""
                floodplain_base = ""
                total_area = ""

                if cross_section.extra_total_volume > 0:
                    b_summerdike = "1"
                    crest_level = str(cross_section.crest_level)
                    total_area = str(cross_section.extra_total_area)

                    # check for nan, because a channel with only one roughness value (ideal case) will not
                    # have this value
                    if not np.isnan(cross_section.floodplain_base):
                        floodplain_base = str(cross_section.floodplain_base)
                    else:
                        floodplain_base = str(
                            cross_section.crest_level,
                        )  # virtual summer dike

                fid.write(
                    cross_section.name
                    + ",,"
                    + "meta"
                    + ",,,,"
                    + "ZW"
                    + ","
                    + str(cross_section.branch)
                    + ","
                    + str(cross_section.chainage)
                    + ","
                    + str(cross_section.section_widths.get("main"))
                    + ","
                    + str(cross_section.section_widths.get("floodplain1"))
                    + ","
                    + str(cross_section.section_widths.get("floodplain2"))
                    + ",,"
                    + b_summerdike
                    + ","
                    + crest_level
                    + ","
                    + floodplain_base
                    + ","
                    + total_area
                    + ","
                    + total_area
                    + ",,,,,,"
                    + "\n",
                )

                # this is to avoid the unique z-value error in sobek, the added 'error' depends on the total_width,
                # this is to make sure the order or points is correct
                z_format = "{:.8f}"
                increment = np.array(range(1, cross_section.z.size + 1)) * 1e-5
                z_value = cross_section.z + increment

                # write geometry information
                for index, width in enumerate(cross_section.total_width):
                    flow_width = cross_section.flow_width[index]
                    fid.write(
                        cross_section.name
                        + ",,"
                        + "geom"
                        + ","
                        + z_format.format(z_value[index])
                        + ","
                        + str(width)
                        + ","
                        + str(flow_width)
                        + ",,,,,,,,,,,,,,"
                        + "\n",
                    )
            except:
                self.set_logger_message(
                    f"Cross-section {cross_section.name} not written to sobek format cross-section file",
                    "error",
                )

    def _write_roughness_sobek3(
        self,
        fid: TextIOWrapper,
        cross_sections: list[CrossSection],
    ) -> None:
        # note, the chainage is currently set to the X-coordinate of the cross-section (straight channel)
        # note, the channel naming strategy must be discussed, currently set to 'Channel' for all cross-sections

        # write header
        fid.write(
            "Name,Chainage,RoughnessType,SectionType,Dependance,Interpolation,Pos/neg,R_pos_constant,Q_pos,R_pos_f(Q),H_pos,R_pos__f(h),R_neg_constant,Q_neg,R_neg_f(Q),H_neg,R_neg_f(h)\n",
        )

        sections = np.unique(
            [s for css in cross_sections for s in css.friction_tables],
        )

        for section in sections:
            for cross_section in cross_sections:
                if section in list(cross_section.friction_tables.keys()):
                    if section == "main":
                        table = cross_section.friction_tables[section]
                        plain = "Main"
                    elif section == "floodplain1":
                        table = cross_section.friction_tables[section]
                        plain = "FloodPlain1"
                    elif section == "floodplain2":
                        table = cross_section.friction_tables[section]
                        plain = "FloodPlain2"
                    else:
                        err_msg = f"Unknown section name: {section}"
                        raise ValueError(err_msg)

                    for level, friction in zip(table.level, table.friction):
                        fid.write(
                            str(cross_section.branch)
                            + ","
                            + str(cross_section.chainage)
                            + ","
                            + "Chezy"
                            + ","
                            + plain
                            + ","
                            + "Waterlevel"
                            + ","
                            + "Linear"
                            + ","
                            + "Same"
                            + ",,,,"
                            + str(level)
                            + ","
                            + str(friction)
                            + ",,,,,"
                            + "\n",
                        )
