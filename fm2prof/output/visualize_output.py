"""Class for visualizing model output data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Generator

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from fm2prof.common import FM2ProfBase
from fm2prof.output.plot_styles import PlotStyles

if TYPE_CHECKING:
    from io import TextIOWrapper
    from logging import Logger

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend


class VisualiseOutput(FM2ProfBase):
    """Visaulise output class."""

    __cssdeffile = "CrossSectionDefinitions.ini"
    __volumefile = "volumes.csv"
    __rmainfile = "roughness-Main.ini"
    __rfp1file = "roughness-FloodPlain1.ini"

    def __init__(
        self,
        output_directory: str,
        logger: Logger | None = None,
    ) -> None:
        """Instantiate a VisualiseOutput object."""
        super().__init__(logger=logger)

        if not logger:
            self._create_logger()
        self.output_dir = Path(output_directory)
        self.fig_dir = self._generate_output_dir()
        self._set_files()
        self._ref_geom_y = []
        self._ref_geom_tw = []
        self._ref_geom_fw = []

        self.set_logger_message(f"Using {self.fig_dir} as output directory for figures")

        # initiate plotstyle
        PlotStyles.apply()

    @property
    def branches(self) -> tuple[np.ndarray, np.ndarray]:
        """Get branches."""
        css_names = [self.split_css(css.get("id")) for css in self.cross_sections]
        branches = np.unique([i[2] for i in css_names])
        contiguous_branches = np.unique([b.split("_")[0] for b in branches])
        return branches, contiguous_branches

    @property
    def number_of_cross_sections(self) -> int:
        """Get number of cross sections."""
        return len(list(self.cross_sections))

    @property
    def cross_sections(self) -> Generator[dict, None, None]:
        """Generator to loop through all cross-sections in definition file.

        Example use:

        >>> for css in visualiser.cross_sections:
        >>>     visualiser.make_figure(css)
        """
        csslist = self._read_css_def_file()
        yield from csslist

    def figure_roughness_longitudinal(self, branch: str) -> None:
        """Get figure of longitudinal roughness.

        Assumes the following naming convention:
        [branch]_[optional:branch_order]_[chainage]
        """
        output_dir = self.fig_dir.joinpath("roughness")
        output_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(1, figsize=(12, 5))

        css = self.get_cross_sections_for_branch(branch)

        chainage = []
        minmax = []
        for cross_section in css:
            chainage.append(cross_section[1])
            roughness = self.get_roughness_info_for_css(cross_section[0])[1]
            minmax.append([min(roughness), max(roughness)])

        chainage = np.array(chainage) * 1e-3
        minmax = np.array(minmax)
        ax.plot(chainage, minmax[:, 0], label="minimum")
        ax.plot(chainage, minmax[:, 1], label="maximum")
        ax.set_ylabel("Ruwheid (Chezy)")
        ax.set_xlabel("Afstand [km]")
        ax.set_title(branch)
        fig, lgd = self._set_plot_style(fig, use_legend=True)
        plt.savefig(
            output_dir.joinpath(f"roughness_longitudinal_{branch}.png"),
            bbox_extra_artists=[lgd],
            bbox_inches="tight",
        )

    def get_cross_sections_for_branch(self, branch: str) -> tuple[str, float, str]:
        """Get cross sections for branch name.

        Args:
            branch (str): branch name.


        Returns:
            tuple[str, float, str]:

        """
        if branch not in self.branches:
            err_msg = f"Branch {branch} not in known branches: {self.branches}"
            raise KeyError(err_msg)

        css_list = [self.split_css(css.get("id")) for css in self.cross_sections]
        branches, contiguous_branches = self.branches
        branch_list = []
        sub_branches = np.unique([b for b in branches if b.startswith(branch)])
        running_chainage = 0
        for i, sub_branch in enumerate(sub_branches):
            sublist = self.get_css_for_branch(css_list, sub_branch)
            if i > 0:
                running_chainage += self.get_css_for_branch(css_list, sub_branches[i - 1])[-1][1]
            branch_list.extend([(s[0], s[1] + running_chainage, s[2]) for s in sublist])

        return branch_list

    @staticmethod
    def split_css(name: str) -> tuple[str, float, str]:
        """Split cross section name."""
        chainage = float(name.split("_")[-1])
        branch = "_".join(name.split("_")[:-1])
        return (name, chainage, branch)

    @staticmethod
    def get_css_for_branch(css_list: list[tuple[str, float, str]], branchname: str) -> list[tuple[str, float, str]]:
        """Get cross section for given branch name."""
        return [c for c in css_list if c[2].startswith(branchname)]

    def get_roughness_info_for_css(self, cssname: str, rtype: str = "roughnessMain") -> tuple[list, list]:
        """Open roughness file and reads information for a given cross-section name."""
        levels = None
        values = None
        with self.files[rtype].open("r") as f:
            cssbranch, csschainage = self._parse_cssname(cssname)
            for line in f:
                if line.strip().lower() == "[branchproperties]" and self._get_value_from_line(f).lower() == cssbranch:
                    [f.readline() for i in range(3)]
                    levels = list(map(float, self._get_value_from_line(f).split()))
                if (
                    line.strip().lower() == "[definition]"
                    and self._get_value_from_line(f).lower() == cssbranch
                    and float(self._get_value_from_line(f).lower()) == csschainage
                ):
                    values = list(map(float, self._get_value_from_line(f).split()))
        return levels, values

    def get_volume_info_for_css(self, cssname: str) -> dict:
        """Get volume info for cross section."""
        column_names = [
            "z",
            "2D_total_volume",
            "2D_flow_volume",
            "2D_wet_area",
            "2D_flow_area",
            "1D_total_volume_sd",
            "1D_total_volume",
            "1D_flow_volume_sd",
            "1D_flow_volume",
            "1D_total_width",
            "1D_flow_width",
        ]
        cssdata = {}
        for column in column_names:
            cssdata[column] = []

        with self.files["volumes"].open("r") as f:
            for line in f:
                values = line.strip().split(",")
                if values[0] == cssname:
                    for i, column in enumerate(column_names):
                        cssdata[column].append(float(values[i + 1]))

        return cssdata

    def get_cross_section_by_id(self, css_id: str) -> dict | None:
        """Get cross-section information given an id.

        Args:
            css_id (str): cross-section name

        """
        csslist = self._read_css_def_file()
        for css in csslist:
            if css.get("id") == css_id:
                return css
        return None

    def figure_cross_section(
        self,
        css: dict,
        reference_geometry: tuple = (),
        reference_roughness: tuple = (),
        *,
        save_to_file: bool = True,
        overwrite: bool = False,
    ) -> Figure:
        """Get a figure of the cross section.

        Args:
            css (dict): cross section dict
            reference_geometry (tuple, optional): tuple of reference . Defaults to ().
            reference_roughness (tuple, optional): _description_. Defaults to ().
            save_to_file (bool, optional): Save the figure to file. Defaults to True.
            overwrite (bool, optional): Overwrite the figure. Defaults to False.

        Returns:
            Figure: _description_

        """
        output_dir = self.fig_dir.joinpath("cross_sections")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir.joinpath(f"{css['id']}.png")
        if output_file.is_file() and not overwrite:
            self.set_logger_message("file already exists", "debug")
            return None
        try:
            fig = plt.figure(figsize=(8, 12))
            gs = fig.add_gridspec(2, 2)
            axs = [
                fig.add_subplot(gs[0, :]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
            ]

            self._plot_geometry(css, axs[0], reference_geometry)
            self._plot_volume(css, axs[1])
            self._plot_roughness(css, axs[2], reference_roughness)

            fig, lgd = self._set_plot_style(fig)

            if save_to_file:
                plt.savefig(
                    output_file,
                    bbox_extra_artists=[lgd],
                    bbox_inches="tight",
                )
            else:
                return fig

        except Exception as e:
            self.set_logger_message(f"error processing: {css['id']} {e!s}", "error")
            return None

        finally:
            plt.close()

    def plot_cross_sections(self) -> None:
        """Plot figures for all cross-sections in project.

        Outputs to output directory of project.
        """
        pbar = tqdm.tqdm(total=self.number_of_cross_sections)
        self.start_new_log_task("Plotting cross-secton figures", pbar=pbar)

        for css in self.cross_sections:
            self.figure_cross_section(css, pbar=pbar)
            pbar.update(1)

        self.finish_log_task()

    def _generate_output_dir(self) -> Path:
        """Create a new directory in the output map to store figures for each cross-section.

        Arguments:
            output_map - path to fm2prof output directory

        Returns:
            png images saved to file

        """
        figdir = self.output_dir.joinpath("figures")
        figdir.mkdir(parents=True, exist_ok=True)
        return figdir

    def _set_files(self) -> None:
        self.files = {
            "css_def": self.output_dir / self.__cssdeffile,
            "volumes": self.output_dir / self.__volumefile,
            "roughnessMain": self.output_dir / self.__rmainfile,
            "roughnessFP1": self.output_dir / self.__rfp1file,
        }

    def _get_value_from_line(self, f: TextIOWrapper) -> str:
        return f.readline().strip().split("=")[1].strip()

    def _read_css_def_file(self) -> list[dict]:
        csslist = []

        with self.files.get("css_def").open("r") as f:
            for line in f:
                if line.lower().strip() == "[definition]":
                    css_id = f.readline().strip().split("=")[1]
                    [f.readline() for i in range(3)]
                    css_levels = list(map(float, self._get_value_from_line(f).split()))
                    css_fwidth = list(map(float, self._get_value_from_line(f).split()))
                    css_twidth = list(map(float, self._get_value_from_line(f).split()))
                    css_sdcrest = float(self._get_value_from_line(f))
                    css_sdflow = float(self._get_value_from_line(f))
                    css_sdtotal = float(self._get_value_from_line(f))
                    css_sdbaselevel = float(self._get_value_from_line(f))
                    css_mainsectionwidth = float(self._get_value_from_line(f))
                    css_fp1sectionwidth = float(self._get_value_from_line(f))

                    css = {
                        "id": css_id.strip(),
                        "levels": css_levels,
                        "flow_width": css_fwidth,
                        "total_width": css_twidth,
                        "SD_crest": css_sdcrest,
                        "SD_flow_area": css_sdflow,
                        "SD_total_area": css_sdtotal,
                        "SD_baselevel": css_sdbaselevel,
                        "mainsectionwidth": css_mainsectionwidth,
                        "fp1sectionwidth": css_fp1sectionwidth,
                    }
                    csslist.append(css)

        return csslist

    def _set_plot_style(self, *args: tuple, **kwargs: dict) -> tuple[Figure, Legend]:
        """Set plot style.

        TODO: add preference to switch styles or
        inject own style
        """
        return PlotStyles.apply(*args, **kwargs)

    def _plot_geometry(self, css: dict, ax: Axes, reference_geometry: list | None = None) -> None:
        # Get data
        tw = np.append([0], np.array(css["total_width"]))
        fw = np.append([0], np.array(css["flow_width"]))
        levels = np.append(css["levels"][0], np.array(css["levels"]))
        mainsectionwidth = css["mainsectionwidth"]
        fp1sectionwidth = css["fp1sectionwidth"]

        # Get the water level where water level independent computation takes over
        # this is the lowest level where there is 2D information on volumes
        z_waterlevel_independent = self._get_lowest_water_level_in_2d(css)

        # Plot cross-section geometry
        for side in [-1, 1]:
            h = ax.fill_betweenx(levels, side * fw / 2, side * tw / 2, color="#44B1D5AA", hatch="////")
            ax.plot(side * tw / 2, levels, "-k")
            ax.plot(side * fw / 2, levels, "--k")

        # Plot roughness section width
        ax.plot(
            [-0.5 * mainsectionwidth, 0.5 * mainsectionwidth],
            [min(levels) - 0.25] * 2,
            "-",
            linewidth=2,
            color="red",
            label="Main section",
        )
        ax.plot(
            [
                -0.5 * (mainsectionwidth + fp1sectionwidth),
                0.5 * (mainsectionwidth + fp1sectionwidth),
            ],
            [min(levels) - 0.25] * 2,
            "--",
            color="red",
            label="Floodplain section",
        )

        # Plot water level indepentent line
        ax.plot(
            tw - 0.5 * max(tw),
            [z_waterlevel_independent] * len(levels),
            linestyle="--",
            color="m",
            label="Lowest water level in 2D",
        )

        h.set_label("Storage")

        ax.set_title(css["id"])
        ax.set_xlabel("[m]")
        ax.set_ylabel("[m]")

        if reference_geometry:
            ax.plot(
                reference_geometry[1],
                reference_geometry[0],
                "--r",
                label="Reference geometry",
            )

        # Plot crest level height
        sd_info = self._get_sd_plot_info(css)

        ax.plot(
            [-tw[-1] / 2, tw[-1] / 2],
            [sd_info.get("crest")] * 2,
            linestyle=sd_info.get("linestyle"),
            linewidth=1,
            color="orange",
            label=sd_info.get("label"),
        )

    def _plot_volume(self, css: dict, ax: Axes) -> None:
        # Get data
        vd = self.get_volume_info_for_css(css["id"])
        z_waterlevel_independent = self._get_lowest_water_level_in_2d(css)

        # Plot 1D volumes
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_total_volume_sd"],
            color="#24A493",
            label="1D Total Volume (incl. SD)",
        )
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_total_volume"],
            color="#108A7A",
            label="1D Total Volume (excl. SD)",
        )
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_flow_volume"],
            color="#209AB4",
            label="1D Flow Volume (incl. SD)",
        )
        ax.fill_between(
            vd["z"],
            0,
            vd["1D_flow_volume"],
            color="#0C6B7F",
            label="1D Flow Volume (excl. SD)",
        )
        # Plot 2D volume
        ax.plot(vd["z"], vd["2D_total_volume"], "--k", label="2D Total Volume")
        ax.plot(vd["z"], vd["2D_flow_volume"], "-.k", label="2D Flow Volume")

        # Plot lowest point in 2D
        ax.plot(
            [z_waterlevel_independent] * len(vd["2D_total_volume"]),
            vd["2D_total_volume"],
            linestyle="--",
            linewidth=1,
            color="m",
            label="Lowest water level in 2D",
        )

        # Plot SD crest
        sd_info = self._get_sd_plot_info(css)
        ax.plot(
            [sd_info.get("crest")] * len(vd["2D_total_volume"]),
            vd["2D_total_volume"],
            linestyle=sd_info.get("linestyle"),
            linewidth=1,
            color="orange",
            label=sd_info.get("label"),
        )

        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xlim([min(vd["z"]), max(vd["z"])])
        ax.set_title("Volume graph")
        ax.set_xlabel("Water level [m]")
        ax.set_ylabel("Volume [m$^3$]")

    def _plot_roughness(self, css: dict, ax: Axes, reference_roughness: tuple) -> None:
        levels, values = self.get_roughness_info_for_css(css["id"], rtype="roughnessMain")
        try:
            ax.plot(levels, values, label="Main channel")
        except:
            pass

        try:
            levels, values = self.get_roughness_info_for_css(css["id"], rtype="roughnessFP1")
            if levels is not None and values is not None:
                ax.plot(levels, values, label="Floodplain1")
        except FileNotFoundError:
            pass

        if reference_roughness:
            ax.plot(
                reference_roughness[0],
                reference_roughness[1],
                "--r",
                label="Reference friction",
            )

        # Limit x axis to min and maximum level in cross-section
        ax.set_xlim(min(css["levels"]), max(css["levels"]))
        ax.set_title("Roughness")
        ax.set_xlabel("Water level [m]")
        ax.set_ylabel("Chezy coefficient [m$^{1/2}$/s]")

    @staticmethod
    def _get_sd_plot_info(css: dict) -> dict:
        v = np.append(css["levels"][0], np.array(css["levels"]))
        z_crest_level = css["SD_crest"]
        if z_crest_level <= max(v):
            if z_crest_level >= min(v):
                sd_linestyle = "--"
                sd_label = "SD Crest Level"
            else:
                z_crest_level = min(v)
                sd_linestyle = "-"
                sd_label = "SD Crest Level (cropped)"
        else:
            z_crest_level = max(v)
            sd_linestyle = "-"
            sd_label = "SD Crest Level (cropped)"

        return {"linestyle": sd_linestyle, "label": sd_label, "crest": z_crest_level}

    def _get_lowest_water_level_in_2d(self, css: dict) -> float:
        vd = self.get_volume_info_for_css(css["id"])
        index_waterlevel_independent = np.argmax(~np.isnan(vd.get("2D_total_volume")))
        return vd.get("z")[index_waterlevel_independent]

    def _parse_cssname(self, cssname: str) -> tuple[str, float]:
        """Return name of branch and chainage."""
        branch, chainage = cssname.rsplit("_", 1)  # rsplit prevents error if branchname contains _
        chainage = round(float(chainage), 2)

        return branch, chainage
