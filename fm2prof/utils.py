import ast
import locale
import os
import re
import warnings
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging import Logger
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.ticker import MultipleLocator
from netCDF4 import Dataset
from pandas.plotting import register_matplotlib_converters
from scipy.interpolate import interp1d

from fm2prof import Project
from fm2prof.common import FM2ProfBase

register_matplotlib_converters()

FigureOutput = namedtuple("FigureOutput", ["fig", "axes", "legend"])
StyleGuide = namedtuple(
    "StyleGuide", ["font", "major_grid", "minor_grid", "spine_width"]
)

COLORSCHEMES = {
    "Deltares": ["#000000", "#00cc96", "#0d38e0"],
    "Koeln": [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E444",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ],  # https://jfly.uni-koeln.de/color/
    "PaulTolVibrant": [
        "#0077BB",
        "#33BBEE",
        "#009988",
        "#EE7733",
        "#CC3311",
        "#EE3377",
        "#BBBBBB",
    ],
}


class GenerateCrossSectionLocationFile(FM2ProfBase):
    """
    Builds a cross-section input file for FM2PROF from a SOBEK 3 DIMR network definition file.

    The distance between cross-section is computed from the differences between the offsets/chainages.
    The beginning and end point of each branch are treated as half-distance control volumes.

    It supports an optional :ref:`branchRuleFile.

    Use as a function i.e. this code will generate a cross-section location file:

    >>> GenerateCrossSectionLocationFile(**input)

    Parameters

        networkdefinitionfile: path to NetworkDefinitionFile.ini

        crossectionlocationfile: path to the desired output file

        branchrulefile: OPTIONAL path to a branchrulefile




    branchrulefile
    ^^^^^^^^^^^^^^
    This file may be used to exclude certain computational points from being
    used as the location of a cross-section. This is particularily useful
    when smaller branches connect to a major branch.

    The branchrule file is a comma-seperates file with the following syntaxt:

    .. code-block:: shell

        branch,rules

    Here, `branch` is the name of the branch and `rules` are rules for exclusion

    Supported general rules are:

    - onlyFirst: only keep the first cross-section, and exclude all others
    - onlyLast: only keep the last cross-section, and exclude all others
    - onlyEdges: only keep the first and last cross-section, and exclude all others
    - ignoreFirst: exclude the first cross-section on a branch
    - ignoreLast: exclude the last cross-section on a branch
    - ignoreEdges: exclude the first and last cross-section on a branch
    - noRule: use to not use any of the above rules

    Additionally, specific cross-sections can be excluded by id. For example:


    .. code-block:: shell

        Channel1, noRule, channel_1_350.000

    In this case, the computational point with name `channel_1_350.000` will
    not be used as the location of a cross-section.

    Rules and individual exclusions can be mixed, e.g.:

    .. code-block:: shell

        Channel1, ignoreLast, channel_1_350.000

    """

    def __init__(
        self,
        networkdefinitionfile: Union[str, Path],
        crossectionlocationfile: Union[str, Path],
        branchrulefile: Optional[Union[str, Path]] = "",
    ):
        super().__init__()

        networkdefinitionfile, crossectionlocationfile, branchrulefile = map(
            Path, [networkdefinitionfile, crossectionlocationfile, branchrulefile]
        )

        required_files = (networkdefinitionfile.is_file(),)
        if not all(required_files):
            raise FileNotFoundError

        self._networkdeffile_to_input(
            networkdefinitionfile, crossectionlocationfile, branchrulefile
        )

    def _parse_NetworkDefinitionFile(
        self, networkdefinitionfile: Path, branchrules: Optional[Dict] = None
    ) -> Dict:
        """
        Output:

            x,y : coordinates of cross-section
            cid : name of the cross-section
            cdis: half-way distance between cross-section points on either side
            bid : name of the branch
            coff:  chainage of cross-section on branch

        """
        if not branchrules:
            branchrules = {}

        # Open network definition file, for each branch extract necessary info
        x = []  # x-coordinate of cross-section centre
        y = []  # y-coordinate of cross-section centre
        cid = []  # id of cross-section
        bid = []  # id of 1D branch
        coff = []  # offset of cross-section on 1D branch ('chainage')
        cdis = []  # distance of 1D branch influenced by crosss-section ('vaklengte')

        with open(networkdefinitionfile, "r") as f:
            for line in f:
                if line.strip().lower() == "[branch]":
                    branchid = f.readline().split("=")[1].strip()
                    xlength = 0
                    for i in range(10):
                        bline = f.readline().strip().lower().split("=")
                        if bline[0].strip() == "gridpointx":
                            xtmp = list(map(float, bline[1].split()))
                        elif bline[0].strip() == "gridpointy":
                            ytmp = list(map(float, bline[1].split()))
                        elif bline[0].strip() == "gridpointids":
                            cidtmp = bline[1].split(";")
                        elif bline[0].strip() == "gridpointoffsets":
                            cofftmp = list(map(float, bline[1].split()))

                            # compute distance between control volumes
                            cdistmp = np.append(np.diff(cofftmp) / 2, [0]) + np.append(
                                [0], np.diff(cofftmp) / 2
                            )

                    cdistmp = list(cdistmp)
                    # Append branchids
                    bidtmp = [branchid] * len(xtmp)

                    # strip cross-section ids
                    cidtmp = [c.strip() for c in cidtmp]

                    # Correct end points (: at end of branch, gridpoints of this branch and previous branch
                    # occupy the same position, which does not go over well with fm2profs classification algo)
                    offset = 1
                    xtmp[0] = np.interp(offset, cofftmp, xtmp)
                    ytmp[0] = np.interp(offset, cofftmp, ytmp)
                    offset = cofftmp[-1] - 1
                    xtmp[-1] = np.interp(offset, cofftmp, xtmp)
                    ytmp[-1] = np.interp(offset, cofftmp, ytmp)

                    # Apply Branchrules
                    if branchid in branchrules:
                        rule = branchrules[branchid].get("rule")
                        exceptions = branchrules[branchid].get("exceptions")
                        if rule:
                            (
                                xtmp,
                                ytmp,
                                cidtmp,
                                cdistmp,
                                bidtmp,
                                cofftmp,
                            ) = self._applyBranchRules(
                                rule, xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp
                            )
                        if exceptions:
                            (
                                xtmp,
                                ytmp,
                                cidtmp,
                                cdistmp,
                                bidtmp,
                                cofftmp,
                            ) = self._applyBranchExceptions(
                                exceptions, xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp
                            )
                        c = len(xtmp)
                        for ic in xtmp, ytmp, cidtmp, cdistmp, bidtmp, cofftmp:
                            if len(ic) != c:
                                raise ValueError

                    # Append all points
                    x.extend(xtmp)
                    y.extend(ytmp)
                    cid.extend(cidtmp)
                    cdis.extend(cdistmp)
                    bid.extend(bidtmp)
                    coff.extend(cofftmp)

        return dict(x=x, y=y, css_id=cid, css_len=cdis, branch_id=bid, css_offset=coff)

    def _networkdeffile_to_input(
        self,
        networkdefinitionfile: Path,
        crossectionlocationfile: Path,
        branchrulefile: Path,
    ):
        branchrules: Dict = {}

        if branchrulefile.is_file():
            branchrules = self._parseBranchRuleFile(branchrulefile)

        network_dict = self._parse_NetworkDefinitionFile(
            networkdefinitionfile, branchrules
        )

        self._writeCrossSectionLocationFile(crossectionlocationfile, network_dict)

    def _applyBranchExceptions(self, exceptions, x, y, cid, cdis, bid, coff):
        for exc in exceptions:
            if exc not in cid:
                self.set_logger_message(f"{exc} not found in branch", "error")
                continue

        pop_indices = [cid.index(exc) for exc in exceptions]

        for pop_index in sorted(pop_indices, reverse=True):
            if pop_index == 0:
                (
                    x,
                    y,
                    cid,
                    cdis,
                    bid,
                    coff,
                ) = self._applyBranchRules("ignorefirst", x, y, cid, cdis, bid, coff)
            elif pop_index == len(x) - 1:
                (
                    x,
                    y,
                    cid,
                    cdis,
                    bid,
                    coff,
                ) = self._applyBranchRules("ignorelast", x, y, cid, cdis, bid, coff)
            else:
                # the distance of the popped value is divided over the two on aither side. 
                cdis[pop_index - 1] += cdis[pop_index] / 2
                cdis[pop_index + 1] += cdis[pop_index] / 2
                
                # then, pop the value
                for l in [x, y, cid, cdis, bid, coff]:
                    l.pop(pop_index)
                

        return x, y, cid, cdis, bid, coff

    def _applyBranchRules(
        self,
        rule: str,
        x: float,
        y: float,
        cid: float,
        cdis: float,
        bid: str,
        coff: float,
    ):
        # bfunc: what points to pop (remove from list)
        bfunc = {
            "norule": lambda x: x,
            "onlyedges": lambda x: [
                x[0],
                x[-1],
            ],  # only keep the 2 cross-section on either end of the branch
            "ignoreedges": lambda x: x[
                1:-1
            ],  # keep everything except 2 css on either end of the branch
            "ignorelast": lambda x: x[:-1],  # keep everything except last css on branch
            "ignorefirst": lambda x: x[
                1:
            ],  # keep everything except first css on branch
            "onlyfirst": lambda x: [x[0]],  # keep only the first css on branch
            "onlylast": lambda x: [x[-1]],  # keep only the last css on branch
        }
        # disfunc: how to modify lengths
        disfunc = {
            "onlyedges": lambda x: [sum(x) / 2] * 2,
            "ignoreedges": lambda x: [sum(x[:2]), *x[2:-2], sum(x[-2:])],
            "ignorelast": lambda x: [*x[:-2], sum(x[-2:])],
            "ignorefirst": lambda x: [sum(x[:2]), *x[2:]],
            "onlyfirst": lambda x: [sum(x)],
            "onlylast": lambda x: [sum(x)],
            "norule": lambda x: x,
        }

        try:
            bf = bfunc[rule.lower().strip()]
            df = disfunc[rule.lower().strip()]
            return bf(x), bf(y), bf(cid), df(cdis), bf(bid), bf(coff)
        except KeyError:
            self.set_logger_message(
                f"'{rule}' is not a known branchrules. Known rules are: {list(bfunc.keys())}",
                "error",
            )

    def _parseBranchRuleFile(
        self, branchrulefile: Path, delimiter: str = ","
    ) -> Dict[str, Dict]:
        """
        Parses the branchrule file which is a delimited file (comma by default)
        """
        branchrules: dict = {}
        with open(branchrulefile, "r") as f:
            lines = [line.strip().split(delimiter) for line in f if len(line) > 1]

        for line in lines:
            branch: str = line[0].strip()
            rule: str = line[1].strip()
            exceptions: List = []
            if len(line) > 2:
                exceptions = [e.strip() for e in line[2:]]

            branchrules[branch] = dict(rule=rule, exceptions=exceptions)

        return branchrules

    def _writeCrossSectionLocationFile(
        self, crossectionlocationfile: Path, network_dict: Dict
    ):
        """
        List inputs:

        x,y : coordinates of cross-section
        cid : name of the cross-section
        cdis: half-way distance between cross-section points on either side
        bid : name of the branch
        coff:  chainage of cross-section on branch
        """
        x = network_dict.get("x")
        y = network_dict.get("y")
        cid = network_dict.get("css_id")
        cdis = network_dict.get("css_len")
        bid = network_dict.get("branch_id")
        coff = network_dict.get("css_offset")

        with open(crossectionlocationfile, "w") as f:
            f.write("name,x,y,length,branch,offset\n")
            for i in range(len(x)):
                f.write(
                    f"{cid[i]}, {x[i]:.4f}, {y[i]:.4f}, {cdis[i]:.2f}, {bid[i]}, {coff[i]:.2f}\n"
                )


class VisualiseOutput(FM2ProfBase):
    __cssdeffile = "CrossSectionDefinitions.ini"
    __volumefile = "volumes.csv"
    __rmainfile = "roughness-Main.ini"
    __rfp1file = "roughness-FloodPlain1.ini"

    def __init__(
        self,
        output_directory: str,
        figure_type: str = "png",
        overwrite: bool = True,
        logger: Logger = None,
    ):
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
    def branches(self) -> Generator[List[str], None, None]:
        def split_css(name) -> Tuple[str, float, str]:
            chainage = float(name.split("_")[-1])
            branch = "_".join(name.split("_")[:-1])
            return (name, chainage, branch)

        def find_branches(css_list) -> List[str]:
            branches = np.unique([i[2] for i in css_names])
            contiguous_branches = np.unique([b.split("_")[0] for b in branches])
            return branches, contiguous_branches

        css_names = [split_css(css.get("id")) for css in self.cross_sections]
        branches, contiguous_branches = find_branches(css_names)
        return branches, contiguous_branches

    @property
    def number_of_cross_sections(self) -> int:
        return len(list(self.cross_sections))

    @property
    def cross_sections(self) -> Generator[Dict, None, None]:
        """
        Generator to loop through all cross-sections in definition file.

        Example use:

        >>> for css in visualiser.cross_sections:
        >>>     visualiser.make_figure(css)
        """
        csslist = self._readCSSDefFile()
        for css in csslist:
            yield css

    def figure_roughness_longitudinal(self, branch: str):
        """
        Assumes the following naming convention:
        [branch]_[optional:branch_order]_[chainage]
        """
        output_dir = self.fig_dir.joinpath("roughness")
        if not output_dir.is_dir():
            os.mkdir(output_dir)

        fig, ax = plt.subplots(1, figsize=(12, 5))

        css = self.get_cross_sections_for_branch(branch)

        chainage = []
        minmax = []
        for cross_section in css:
            chainage.append(cross_section[1])
            roughness = self.getRoughnessInfoForCss(cross_section[0])[1]
            minmax.append([min(roughness), max(roughness)])

        chainage = np.array(chainage) * 1e-3
        minmax = np.array(minmax)
        ax.plot(chainage, minmax[:, 0], label="minimum")
        ax.plot(chainage, minmax[:, 1], label="maximum")
        ax.set_ylabel("Ruwheid (Chezy)")
        ax.set_xlabel("Afstand [km]")
        ax.set_title(branch)
        fig, lgd = self._SetPlotStyle(fig, use_legend=True)
        plt.savefig(
            output_dir.joinpath(f"roughness_longitudinal_{branch}.png"),
            bbox_extra_artists=[lgd],
            bbox_inches="tight",
        )

    def get_cross_sections_for_branch(self, branch: str) -> Tuple[str, float, str]:
        if branch not in self.branches:
            raise KeyError(f"Branch {branch} not in known branches: {self.branches}")

        def split_css(name) -> Tuple[str, float, str]:
            chainage = float(name.split("_")[-1])
            branch = "_".join(name.split("_")[:-1])
            return (name, chainage, branch)

        def get_css_for_branch(css_list, branchname: str):
            return [c for c in css_list if c[2].startswith(branchname)]

        css_list = [split_css(css.get("id")) for css in self.cross_sections]
        branches, contiguous_branches = self.branches
        branch_list = []
        sub_branches = np.unique([b for b in branches if b.startswith(branch)])
        running_chainage = 0
        for i, sub_branch in enumerate(sub_branches):
            sublist = get_css_for_branch(css_list, sub_branch)
            if i > 0:
                running_chainage += get_css_for_branch(css_list, sub_branches[i - 1])[
                    -1
                ][1]
            branch_list.extend([(s[0], s[1] + running_chainage, s[2]) for s in sublist])

        return branch_list

    def getRoughnessInfoForCss(self, cssname, rtype: str = "roughnessMain"):
        """
        Opens roughness file and reads information for a given cross-section
        name
        """
        levels = None
        values = None
        with open(self.files[rtype], "r") as f:
            cssbranch, csschainage = self._parse_cssname(cssname)
            for line in f:
                if line.strip().lower() == "[branchproperties]":
                    if self._getValueFromLine(f).lower() == cssbranch:
                        [f.readline() for i in range(3)]
                        levels = list(map(float, self._getValueFromLine(f).split()))
                if line.strip().lower() == "[definition]":
                    if self._getValueFromLine(f).lower() == cssbranch:
                        if float(self._getValueFromLine(f).lower()) == csschainage:
                            values = list(map(float, self._getValueFromLine(f).split()))
        return levels, values

    def getVolumeInfoForCss(self, cssname):
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
            cssdata[column] = list()

        with open(self.files["volumes"], "r") as f:
            for line in f:
                values = line.strip().split(",")
                if values[0] == cssname:
                    for i, column in enumerate(column_names):
                        cssdata[column].append(float(values[i + 1]))

        return cssdata

    def get_cross_section_by_id(self, id: str) -> dict:
        """
        Get cross-section information given an id.

        Arguments:
            id (str): cross-section name
        """
        csslist = self._readCSSDefFile()
        for css in csslist:
            if css.get("id") == id:
                return css

    def figure_cross_section(
        self,
        css,
        reference_geometry: tuple = (),
        reference_roughness: tuple = (),
        save_to_file: bool = True,
        overwrite: bool = False,
        pbar: tqdm.std.tqdm = None,
    ) -> None:
        """
        Creates a figure

        Arguments

            css: dictionary containing cross-section information. Obtain with `VisualiseOutput.cross_sections`
                 generator or `VisualiseOutput.get_cross_section_by_id` method.

            reference_geometry (tuple): tuple(list(y), list(z))

            reference_roughness (tuple): tuple(list(z), list(n))

            save_to_file (bool): if true, save figure to VisualiseOutput.fig_dir
                                 if false, returns pyplot figure object

        """
        output_dir = self.fig_dir.joinpath("cross_sections")
        if not output_dir.is_dir():
            os.mkdir(output_dir)

        output_file = output_dir.joinpath(f"{css['id']}.png")
        if output_file.is_file() & ~overwrite:
            self.set_logger_message("file already exists", "debug")
            return
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

            fig, lgd = self._SetPlotStyle(fig)

            if save_to_file:
                plt.savefig(
                    output_file,
                    bbox_extra_artists=[lgd],
                    bbox_inches="tight",
                )
            else:
                return fig

        except Exception as e:
            self.set_logger_message(f"error processing: {css['id']} {str(e)}", "error")
            return None

        finally:
            plt.close()

    def plot_cross_sections(self):
        """Makes figures for all cross-sections in project,
        output to output directory of project"""
        pbar = tqdm.tqdm(total=self.number_of_cross_sections)
        self.start_new_log_task("Plotting cross-secton figures", pbar=pbar)

        for css in self.cross_sections:
            self.figure_cross_section(css, pbar=pbar)
            pbar.update(1)

        self.finish_log_task()

    def _generate_output_dir(self, figure_type: str = "png", overwrite: bool = True):
        """
        Creates a new directory in the output map to store figures for each cross-section

        Arguments:
            output_map - path to fm2prof output directory

        Returns:
            png images saved to file
        """

        figdir = self.output_dir.joinpath("figures")
        if not figdir.is_dir():
            figdir.mkdir(parents=True)
        return figdir

    def _set_files(self):
        self.files = {
            "css_def": os.path.join(self.output_dir, self.__cssdeffile),
            "volumes": os.path.join(self.output_dir, self.__volumefile),
            "roughnessMain": os.path.join(self.output_dir, self.__rmainfile),
            "roughnessFP1": os.path.join(self.output_dir, self.__rfp1file),
        }

    def _getValueFromLine(self, f):
        return f.readline().strip().split("=")[1].strip()

    def _readCSSDefFile(self) -> List[Dict]:
        csslist = list()

        with open(self.files.get("css_def"), "r") as f:
            for line in f:
                if line.lower().strip() == "[definition]":
                    css_id = f.readline().strip().split("=")[1]
                    [f.readline() for i in range(3)]
                    css_levels = list(map(float, self._getValueFromLine(f).split()))
                    css_fwidth = list(map(float, self._getValueFromLine(f).split()))
                    css_twidth = list(map(float, self._getValueFromLine(f).split()))
                    css_sdcrest = float(self._getValueFromLine(f))
                    css_sdflow = float(self._getValueFromLine(f))
                    css_sdtotal = float(self._getValueFromLine(f))
                    css_sdbaselevel = float(self._getValueFromLine(f))
                    css_mainsectionwidth = float(self._getValueFromLine(f))
                    css_fp1sectionwidth = float(self._getValueFromLine(f))

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

    def _SetPlotStyle(self, *args, **kwargs):
        """todo: add preference to switch styles or
        inject own style
        """
        return PlotStyles.apply(*args, **kwargs)

    def _plot_geometry(self, css, ax, reference_geometry=None):
        # Get data
        tw = np.append([0], np.array(css["total_width"]))
        fw = np.append([0], np.array(css["flow_width"]))
        l = np.append(css["levels"][0], np.array(css["levels"]))
        mainsectionwidth = css["mainsectionwidth"]
        fp1sectionwidth = css["fp1sectionwidth"]

        # Get the water level where water level independent computation takes over
        # this is the lowest level where there is 2D information on volumes
        z_waterlevel_independent = self._get_lowest_water_level_in_2D(css)

        # Plot cross-section geometry
        for side in [-1, 1]:
            h = ax.fill_betweenx(
                l, side * fw / 2, side * tw / 2, color="#44B1D5AA", hatch="////"
            )
            ax.plot(side * tw / 2, l, "-k")
            ax.plot(side * fw / 2, l, "--k")

        # Plot roughness section width
        ax.plot(
            [-0.5 * mainsectionwidth, 0.5 * mainsectionwidth],
            [min(l) - 0.25] * 2,
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
            [min(l) - 0.25] * 2,
            "--",
            color="red",
            label="Floodplain section",
        )
        # ax.plot(tw-0.5*max(fp1sectionwidth),[min(l)]*len(l), '--', color='cyan', label='Floodplain section')

        # Plot water level indepentent line
        ax.plot(
            tw - 0.5 * max(tw),
            [z_waterlevel_independent] * len(l),
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

    def _plot_volume(self, css, ax):
        # Get data
        vd = self.getVolumeInfoForCss(css["id"])
        z_waterlevel_independent = self._get_lowest_water_level_in_2D(css)

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

    def _plot_roughness(self, css, ax, reference_roughness):
        levels, values = self.getRoughnessInfoForCss(css["id"], rtype="roughnessMain")
        try:
            ax.plot(levels, values, label="Main channel")
        except:
            pass

        try:
            levels, values = self.getRoughnessInfoForCss(
                css["id"], rtype="roughnessFP1"
            )
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
    def _get_sd_plot_info(css):
        l = np.append(css["levels"][0], np.array(css["levels"]))
        z_crest_level = css["SD_crest"]
        if z_crest_level <= max(l):
            if z_crest_level >= min(l):
                sd_linestyle = "--"
                sd_label = "SD Crest Level"
            else:
                z_crest_level = min(l)
                sd_linestyle = "-"
                sd_label = "SD Crest Level (cropped)"
        else:
            z_crest_level = max(l)
            sd_linestyle = "-"
            sd_label = "SD Crest Level (cropped)"

        return {"linestyle": sd_linestyle, "label": sd_label, "crest": z_crest_level}

    def _get_lowest_water_level_in_2D(self, css):
        vd = self.getVolumeInfoForCss(css["id"])
        index_waterlevel_independent = np.argmax(~np.isnan(vd.get("2D_total_volume")))
        z_waterlevel_independent = vd.get("z")[index_waterlevel_independent]
        return z_waterlevel_independent

    def _parse_cssname(self, cssname):
        """
        returns name of branch and chainage
        """
        branch, chainage = cssname.rsplit(
            "_", 1
        )  # rsplit prevents error if branchname contains _
        chainage = round(float(chainage), 2)

        return branch, chainage


class PlotStyles:
    myFmt = mdates.DateFormatter("%d-%b")
    monthlocator = mdates.MonthLocator(bymonthday=(1, 10, 20))
    daylocator = mdates.DayLocator(interval=5)
    colorscheme = COLORSCHEMES["Koeln"]

    @staticmethod
    def set_locale(localeString: str):
        try:
            locale.setlocale(locale.LC_TIME, localeString)
        except locale.Error:
            # known error on linux fix:
            # export LC_ALL="en_US.UTF-8" & export LC_CTYPE="en_US.UTF-8" & sudo dpkg-reconfigure locales
            print(f"could not set locale to {localeString}")
            pass

    @staticmethod
    def _is_timeaxis(axis) -> bool:
        try:
            label_string = axis.get_ticklabels()[0].get_text().replace("−", "-")
            # if label_string is empty (e.g. because of twin_axis, return false)
            if label_string:
                float(label_string)

        except ValueError:
            return True
        except IndexError:
            return False
        return False

    @classmethod
    def van_veen(
        cls,
        fig: Figure | None = None,
        use_legend: bool = True,
        extra_labels: List | None = None,
        ax_align_legend: plt.Axes | None = None,
    ):
        warnings.warn(
            "This function is deprecated and will be removed on future versions. Use PlotStyle.apply(fig, style='van_veen') instead",
            category=DeprecationWarning,
        )
        cls.apply(
            fig=fig,
            style="van_veen",
            use_legend=use_legend,
            extra_labels=extra_labels,
            ax_align_legend=ax_align_legend,
        )

    @classmethod
    def apply(
        cls,
        fig: Figure | None = None,
        style: str = "sito",
        use_legend: bool = True,
        extra_labels: List | None = None,
        ax_align_legend: plt.Axes | None = None,
    ) -> Tuple[Figure, Legend]:
        styles: Dict[str, StyleGuide] = dict(
            sito=StyleGuide(
                font={"family": "Franca, Arial", "weight": "normal", "size": 16},
                major_grid=dict(
                    visible=True,
                    which="major",
                    linestyle="--",
                    linewidth=1.0,
                    color="#BBBBBB",
                ),
                minor_grid=dict(
                    visible=True,
                    which="minor",
                    linestyle="--",
                    linewidth=1.0,
                    color="#BBBBBB",
                ),
                spine_width=1,
            ),
            van_veen=StyleGuide(
                font={"family": "Bahnschrift", "weight": "normal", "size": 18},
                major_grid=dict(
                    visible=True, which="major", linestyle="-", linewidth=1, color="k"
                ),
                minor_grid=dict(
                    visible=True, which="minor", linestyle="-", linewidth=0.5, color="k"
                ),
                spine_width=2,
            ),
        )

        if style not in styles:
            raise KeyError(f"unknown style {style}. Options are {list(styles.keys())}")

        style_guide: StyleGuide = styles.get(style)

        def initiate() -> None:
            # Set default locale to NL
            # TODO: add localization options (#85)
            PlotStyles.set_locale("nl_NL.UTF-8")

            # Color style
            mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
                color=cls.colorscheme * 3,
                linestyle=["-"] * len(cls.colorscheme)
                + ["--"] * len(cls.colorscheme)
                + ["-."] * len(cls.colorscheme),
            )

            # Font style
            font = style_guide.font

            # not all fonts support the unicode minus, so disable this option
            mpl.rc("font", **font)
            mpl.rcParams["axes.unicode_minus"] = False

        def style_figure(
            fig, use_legend, extra_labels, ax_align_legend
        ) -> Tuple[Figure, Legend] | Tuple[Figure, List] | None:
            if ax_align_legend is None:
                ax_align_legend = fig.axes[0]

            # this forces labels to be generated. Necessary to detect datetimes
            fig.canvas.draw()

            # Set styles for each axis
            legend_title = r"Toelichting"
            handles = list()
            labels = list()

            for ax in fig.axes:
                # Enable grid grid
                ax.grid(**style_guide.major_grid)
                ax.grid(**style_guide.minor_grid)

                for _, spine in ax.spines.items():
                    spine.set_linewidth(style_guide.spine_width)

                if cls._is_timeaxis(ax.xaxis):
                    ax.xaxis.set_major_formatter(cls.myFmt)
                    ax.xaxis.set_major_locator(cls.monthlocator)
                if cls._is_timeaxis(ax.yaxis):
                    ax.yaxis.set_major_formatter(cls.myFmt)
                    ax.yaxis.set_major_locator(cls.monthlocator)

                ax.patch.set_visible(False)
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

            if extra_labels:
                handles.extend(extra_labels[0])
                labels.extend(extra_labels[1])
            fig.tight_layout()
            if use_legend:
                lgd = fig.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.0, ax_align_legend.get_position().y1),
                    bbox_transform=fig.transFigure,
                    edgecolor="k",
                    facecolor="white",
                    framealpha=1,
                    borderaxespad=0,
                    title=legend_title.upper(),
                )

                return fig, lgd
            else:
                return fig, handles, labels

        if not fig:
            return initiate()
        else:
            initiate()
            return style_figure(
                fig=fig,
                use_legend=use_legend,
                extra_labels=extra_labels,
                ax_align_legend=ax_align_legend,
            )


class ModelOutputReader(FM2ProfBase):
    """
    This class provides methods to post-process 1D and 2D data,
    by writing csv files of output locations (observation stations)
    that are both in 1D and 2D. It produces two csv files that
    are input for :meth:`fm2prof.utils.ModelOutputPlotter`

    Example use:

    >>> # initialise & give path to 1D, 2D model resusts
    >>> from fm2prof.utils import ModelOutputReader
    >>> output = ModelOutputReader()
    >>> output.path_flow1d = path_to_dimr_directory
    >>> output.path_flow2d = path_to_nc_file
    >>> # Read and write 1d output to csv
    >>> output.load_flow1d_data()
    >>> #
    >>> output.get_1d2d_map()
    >>> output.load_flow2d_data()
    """

    __fileOutName_F1D_Q = "1D_Q.csv"
    __fileOutName_F1D_H = "1D_H.csv"
    __fileOutName_F2D_Q = "2D_Q.csv"
    __fileOutName_F2D_H = "2D_H.csv"

    _key_1D_Q_name = "observation_id"
    _key_1D_Q = "water_discharge"
    _key_1D_time = "time"
    _key_1D_H_name = "observation_id"
    _key_1D_H = "water_level"

    _key_2D_Q_name = "cross_section_name"
    _key_2D_Q = "cross_section_discharge"
    _key_2D_time = "time"
    _key_2D_H_name = "station_name"
    _key_2D_H = "waterlevel"
    __fileOutName_1D2DMap = "map_1d_2d.csv"

    _time_fmt = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        logger=None,
        start_time: datetime | None = None,
        stop_time: datetime | None = None,
    ):
        super().__init__(logger=logger)

        self._path_out: Path = Path(".")
        self._path_flow1d: Path = Path(".")
        self._path_flow2d: Path = Path(".")

        self._data_1D_Q: pd.DataFrame = None
        self._time_offset_1d: int = 0

        self._start_time: datetime | None = start_time
        self._stop_time: datetime | None = stop_time

    @property
    def start_time(self) -> Union[datetime, None]:
        """if defined, used to mask data"""
        return self._start_time

    @start_time.setter
    def start_time(self, input_time: datetime):
        if isinstance(input_time, datetime):
            self._start_time = input_time

    @property
    def stop_time(self) -> Union[datetime, None]:
        """if defined, used to mask data"""
        return self._stop_time

    @stop_time.setter
    def stop_time(self, input_time: datetime):
        if isinstance(input_time, datetime):
            self._stop_time = input_time

    @property
    def path_flow1d(self):
        return self._path_flow1d

    @path_flow1d.setter
    def path_flow1d(self, path: Union[Path, str]):
        # Verify path is dir
        assert Path(path).is_file()
        # set attribute
        self._path_flow1d = Path(path)

    @property
    def path_flow2d(self):
        return self._path_flow2d

    @path_flow2d.setter
    def path_flow2d(self, path: Union[Path, str]):
        # Verify path is file
        assert Path(path).is_file()
        # set attribute
        self._path_flow2d = Path(path)

    def load_flow1d_data(self):
        """
        Loads 'observations.nc' and outputs to csv file

        .. note::
            Path to the 1D model must first be set by using
            >>> ModelOutputReader.path_flow1d = path_to_dir_that_contains_dimr_xml
        """
        if self.file_1D_Q.is_file() & self.file_1D_H.is_file():
            self._data_1D_Q = pd.read_csv(
                self.file_1D_Q,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self._data_1D_H = pd.read_csv(
                self.file_1D_H,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self.set_logger_message("Using existing flow1d csv files")
        else:
            self.set_logger_message("Importing from NetCDF")
            self._data_1D_H, self._data_1D_Q = self._import_1Dobservations()
            self.set_logger_message("Writing to CSV (waterlevels)")
            self._data_1D_H.to_csv(self.file_1D_H)
            self.set_logger_message("Writing to CSV (discharge)")
            self._data_1D_Q.to_csv(self.file_1D_Q)

    def load_flow2d_data(self):
        """
        Loads 2D output file (netCDF, must contain observation point results),
        matches to 1D result, output to csv

        .. note::
            Path to the 2D model output
            >>> ModelOutputReader.path_flow2d = path_to_netcdf_file
        """
        if self.file_2D_Q.is_file() & self.file_2D_H.is_file():
            self._data_2D_Q = pd.read_csv(
                self.file_2D_Q,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self._data_2D_H = pd.read_csv(
                self.file_2D_H,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self.set_logger_message("Using existing flow2d csv files")
        else:
            # write to file
            self._import_2Dobservations()

            # then load
            self._data_2D_Q = pd.read_csv(
                self.file_2D_Q,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )
            self._data_2D_H = pd.read_csv(
                self.file_2D_H,
                index_col=0,
                parse_dates=True,
                date_format=self._time_fmt,
            )

    def get_1d2d_map(self):
        """Writes a map between stations in 1D and stations in 2D. Matches based on identical characters in first nine slots"""
        if self.file_1D2D_map.is_file():
            self.set_logger_message("using existing 1d-2d map")
            return
        else:
            self._get_1d2d_map()

    def read_all_data(self) -> None:
        """ """
        self.load_flow1d_data()
        self.get_1d2d_map()
        self.load_flow2d_data()

    def _dateparser(self, t):
        # DEPRECATED
        return datetime.strptime(t, self._time_fmt)

    @property
    def output_path(self) -> Path:
        return self._path_out

    @output_path.setter
    def output_path(self, new_path: Union[Path, str]):
        newpath = Path(new_path)
        if newpath.is_dir():
            self._path_out = newpath
        else:
            raise ValueError(f"{new_path} is not a directory")

    @property
    def file_1D_Q(self):
        return self.output_path.joinpath(self.__fileOutName_F1D_Q)

    @property
    def file_1D_H(self):
        return self.output_path.joinpath(self.__fileOutName_F1D_H)

    @property
    def file_2D_Q(self):
        return self.output_path.joinpath(self.__fileOutName_F2D_Q)

    @property
    def file_2D_H(self):
        return self.output_path.joinpath(self.__fileOutName_F2D_H)

    @property
    def file_1D2D_map(self):
        return self.output_path.joinpath(self.__fileOutName_1D2DMap)

    @property
    def data_1D_H(self):
        return self._apply_startstop_time(self._data_1D_H)

    @property
    def data_2D_H(self):
        return self._apply_startstop_time(self._data_2D_H)

    @property
    def data_1D_Q(self):
        return self._apply_startstop_time(self._data_1D_Q)

    @property
    def data_2D_Q(self):
        return self._apply_startstop_time(self._data_2D_Q)

    @property
    def time_offset_1d(self):
        return self._time_offset_1d

    @time_offset_1d.setter
    def time_offset_1d(self, seconds: int = 0):
        self._time_offset_1d = seconds

    def _apply_startstop_time(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies stop/start time to data
        """
        if self.stop_time is None:
            self.stop_time = data.index[-1]
        if self.start_time is None:
            self.start_time = data.index[0]

        if self.start_time >= self.stop_time:
            self.set_logger_message(
                "Stop time ({self.stop_time}) should be later than start time ({self.start_time})",
                "error",
            )
            raise ValueError
        if bool(self.start_time) and (self.start_time >= data.index[-1]):
            self.set_logger_message(
                f"Provided start time {self.start_time} is later than last record in data ({data.index[-1]})",
                "error",
            )
            raise ValueError
        if bool(self.stop_time) and (self.stop_time <= data.index[0]):
            self.set_logger_message(
                f"Provided stop time {self.stop_time} is earlier than first record in data ({data.index[0]})",
                "error",
            )
            raise ValueError

        if bool(self.start_time) and bool(self.stop_time):
            return data[
                (data.index >= self.start_time) & (data.index <= self.stop_time)
            ]
        elif bool(self.start_time) and not bool(self.stop_time):
            return data[(data.index >= self.start_time)]
        elif not bool(self.start_time) and bool(self.stop_time):
            return data[data.index <= self.stop_time]
        else:
            return data

    @staticmethod
    def _parse_names(nclist, encoding="utf-8"):
        """Parses the bytestring list of names in netcdf"""
        return [
            "".join([bstr.decode(encoding) for bstr in ncrow]).strip()
            for ncrow in nclist
        ]

    def _import_2Dobservations(self) -> None:
        print("Reading 2D data")
        for nkey, dkey, map_key, fname in zip(
            [self._key_2D_Q_name, self._key_2D_H_name],
            [self._key_2D_Q, self._key_2D_H],
            ["2D_Q", "2D_H"],
            [self.file_2D_Q, self.file_2D_H],
        ):
            with Dataset(self._path_flow2d) as f:
                self.set_logger_message(f"loading 2D data for {map_key}")
                station_map = pd.read_csv(self.file_1D2D_map, index_col=0)
                qnames = self._parse_names(f.variables[nkey][:])
                qdata = f.variables[dkey][:]

                time = self._parse_time(f.variables["time"])
                df = pd.DataFrame(columns=station_map.index, index=time)
                self.set_logger_message("Matching 1D and 2D data")
                for index, station in tqdm.tqdm(
                    station_map.iterrows(), total=len(station_map.index)
                ):
                    # Get index of the current station, or skip if ValueError
                    try:
                        si = qnames.index(station[map_key])
                    except ValueError:
                        continue

                    df[station.name] = qdata[:, si]

                df.to_csv(f"{fname}")

    def _import_1Dobservations(self) -> pd.DataFrame:
        """
        time_offset: offset in seconds
        """
        _file_his = self.path_flow1d

        with Dataset(_file_his) as f:
            names = self._parse_names(
                f.variables[self._key_1D_H_name]
            )  # names are the same for Q in 1D

            time = self._parse_time(f.variables[self._key_1D_time])
            data = f.variables[self._key_1D_H][:]
            dfH = pd.DataFrame(columns=names, index=time, data=data)

            data = f.variables[self._key_1D_Q][:]
            dfQ = pd.DataFrame(columns=names, index=time, data=data)

            # apply index shift
            dfH.index = dfH.index + timedelta(seconds=self.time_offset_1d)
            dfQ.index = dfQ.index + timedelta(seconds=self.time_offset_1d)

            return dfH, dfQ

    def _parse_time(self, timevector: pd.DataFrame):
        """seconds"""
        unit = timevector.units.replace("seconds since ", "").strip()

        try:
            start_time = datetime.strptime(unit, self._time_fmt)
        except ValueError as e:
            if len(e.args) > 0 and e.args[0].startswith("unconverted data remains: "):
                unit = unit[: -(len(e.args[0]) - 26)]
                start_time = datetime.strptime(unit, self._time_fmt)

        return [start_time + timedelta(seconds=i) for i in timevector[:]]

    def _parse_1D_stations(self) -> Generator[str, None, None]:
        """Reads the names of observations stations from 1D model"""
        return list(self._data_1D_H.columns)

    def _get_1d2d_map(self):
        _file_his = self.path_flow2d

        with Dataset(_file_his) as f:
            qnames = self._parse_names(f.variables[self._key_2D_Q_name][:])
            hnames = self._parse_names(f.variables[self._key_2D_H_name][:])

            # get matching names based on first nine characters
            with open(self.file_1D2D_map, "w") as fw:
                fw.write("1D,2D_H,2D_Q\n")
                for n in tqdm.tqdm(list(self._parse_1D_stations())):
                    try:
                        qn = next(x for x in qnames if x.startswith(n[:9]))
                    except StopIteration:
                        qn = ""
                    try:
                        hn = next(x for x in hnames if x.startswith(n[:9]))
                    except StopIteration:
                        hn = ""
                    fw.write(f"{n},{hn},{qn}\n")


class Compare1D2D(ModelOutputReader):
    """
    Utility to compare the results of a 1D and 2D model through
    visualisation and statistical post-processing.

    Note:
        If 2D and 1D netCDF input files are provided, they will first be
        converted to csv files. Once csv files are present, the original
        netCDF files are no longer used. In that case, the arguments
        to `path_1d` and `path_2d` should be `None`.


    Example usage:
        ``` py
        from fm2prof import Project, utils
        project = Project(fr'tests/test_data/compare1d2d/cases/case1/fm2prof.ini')
        plotter = utils.Compare1D2D(project=project,
                                    start_time=datetime(year=2000, month=1, day=5))

        plotter.figure_at_station("NR_919.00")

        ```

    Parameters:
        project: `fm2prof.Project` object.
        path_1d: path to SOBEK dimr directory
        path_2d: path to his nc file
        routes: list of branch abbreviations, e.g. ['NR', 'LK']
        start_time: start time for plotting and analytics. Use this to crop the time to prevent initalisation from affecting statistics.
        stop_time: stop time for plotting and analytics.
        style: `PlotStyles` style
    """

    _routes: List[List[str]] = None

    def __init__(
        self,
        project: Project,
        path_1d: Path | str | None = None,
        path_2d: Path | str | None = None,
        routes: List[List[str]] | None = None,
        start_time: None | datetime = None,
        stop_time: None | datetime = None,
        style: str = "sito",
    ):
        if project:
            super().__init__(
                logger=project.get_logger(), start_time=start_time, stop_time=stop_time
            )
            self.output_path = project.get_output_directory()
        else:
            super().__init__()

        if isinstance(path_1d, (Path, str)) and Path(path_1d).is_file():
            self.path_flow1d = path_1d
        else:
            self.set_logger_message(
                f"1D netCDF file does not exist or is not provided. Input provided: {path_1d}.",
                "debug",
            )
        if isinstance(path_1d, (Path, str)) and Path(path_2d).is_file():
            self.path_flow2d = path_2d
        else:
            self.set_logger_message(
                f"2D netCDF file does not exist or is not provided. Input provided: {path_2d}.",
                "debug",
            )

        # Defaults
        self.routes = routes
        self.statistics = None
        self._data_1D_H: pd.DataFrame = None
        self._data_2D_H: pd.DataFrame = None
        self._data_1D_H_digitized: pd.DataFrame = None
        self._data_2D_H_digitized: pd.DataFrame = None
        self._qsteps = np.arange(0, 100 * np.ceil(18000 / 100), 200)

        # initiate plotstyle
        self._error_colors = ["#7e3e00", "#FF4433", "#d86a00"]
        self._color_error = self._error_colors[1]
        self._color_scheme = COLORSCHEMES["Koeln"]
        self._plotstyle: str = style
        PlotStyles.apply(style=self._plotstyle)

        # set start time
        self.start_time = start_time
        self.stop_time = stop_time

        self.read_all_data()
        self.digitize_data()

        # create output folder
        output_dirs = [
            "figures/longitudinal",
            "figures/discharge",
            "figures/heatmaps",
            "figures/stations",
        ]
        for od in output_dirs:
            try:
                os.makedirs(self.output_path.joinpath(od))
            except FileExistsError:
                pass

    def eval(self):
        """
        does a bunch
        """
        for route in tqdm.tqdm(self.routes):
            self.set_logger_message(f"Making figures for route {route}")
            self.figure_longitudinal_rating_curve(route)
            self.figure_longitudinal_time(route)
            self.heatmap_rating_curve(route)
            self.heatmap_time(route)

        self.set_logger_message(f"Making figures for stations")
        for station in tqdm.tqdm(self.stations(), total=self._data_1D_H.shape[1]):
            self.figure_at_station(station)

    @property
    def routes(self):
        return self._routes

    @routes.setter
    def routes(self, routes):
        if isinstance(routes, list):
            self._routes = routes
        if isinstance(routes, str):
            self._routes = ast.literal_eval(routes)

    @property
    def file_1D_H_digitized(self):
        return self.file_1D_H.parent.joinpath(f"{self.file_1D_H.stem}_digitized.csv")

    @property
    def file_2D_H_digitized(self):
        return self.file_2D_H.parent.joinpath(f"{self.file_2D_H.stem}_digitized.csv")

    @property
    def colorscheme(self):
        return self._colorscheme

    def digitize_data(self):
        if self.file_1D_H_digitized.is_file():
            self.set_logger_message("Using existing digitized file for 1d")
            self._data_1D_H_digitized = pd.read_csv(
                self.file_1D_H_digitized, index_col=0
            )
        else:
            self._data_1D_H_digitized = self._digitize_data(
                self._data_1D_H, self._data_1D_Q, self._qsteps
            )
            self._data_1D_H_digitized.to_csv(self.file_1D_H_digitized)
        if self.file_2D_H_digitized.is_file():
            self.set_logger_message("Using existing digitized file for 2d")
            self._data_2D_H_digitized = pd.read_csv(
                self.file_2D_H_digitized, index_col=0
            )
        else:
            self._data_2D_H_digitized = self._digitize_data(
                self._data_2D_H, self._data_2D_Q, self._qsteps
            )
            self._data_2D_H_digitized.to_csv(self.file_2D_H_digitized)

    def stations(self) -> Generator[str, None, None]:
        for station in self._data_1D_H.columns:
            yield station

    @staticmethod
    def _digitize_data(hdata, qdata, bins) -> pd.DataFrame:
        """Computes the average for a given bin. Use to make Q-H graphs instead of T-H graph"""

        stations = hdata.columns

        C = list()
        rkms = list()
        for i, station in enumerate(stations):
            # rkms.append(float(station.split('_')[1]))
            d = np.digitize(qdata[station], bins)
            C.append([np.nanmean(hdata[station][d == i]) for i, _ in enumerate(bins)])

        # sort = np.argsort(rkms)
        C = np.array(C)  # [sort]
        return pd.DataFrame(columns=stations, index=bins, data=C.T)

    def _names_to_rkms(self, station_names: List[str]) -> List[float]:
        return [
            self._catch_e(lambda: float(i.split("_")[1]), (IndexError, ValueError))
            for i in station_names
        ]

    def _names_to_branches(self, station_names: List[str]) -> List[str]:
        return [
            self._catch_e(lambda: i.split("_")[0], IndexError) for i in station_names
        ]

    def get_route(
        self, route: List[str]
    ) -> Tuple[List[str], List[float], List[Tuple[str, float]]]:
        """returns a sorted list of stations along a route, with rkms"""
        station_names = self._data_2D_H.columns

        # Parse names
        rkms = self._names_to_rkms(station_names)
        branches = self._names_to_branches(station_names)

        # select along route
        routekms = list()
        stations = list()
        lmw_stations = list()

        for stop in route:
            indices = [i for i, b in enumerate(branches) if b == stop]
            routekms.extend([rkms[i] for i in indices])
            stations.extend([station_names[i] for i in indices])
            lmw_stations.extend(
                [
                    (station_names[i], rkms[i])
                    for i in indices
                    if "LMW" in station_names[i]
                ]
            )

        # sort data
        sorted_indices = np.argsort(routekms)
        sorted_stations = [
            stations[i] for i in sorted_indices if routekms[i] is not np.nan
        ]
        sorted_rkms = [routekms[i] for i in sorted_indices if routekms[i] is not np.nan]

        # sort lmw stations
        lmw_stations = [
            lmw_stations[j] for j in np.argsort([i[1] for i in lmw_stations])
        ]
        return sorted_stations, sorted_rkms, lmw_stations

    def statistics_to_file(self, file_path: str = "error_statistics") -> None:
        """
        Creates and output a file `error_statistics.csv', which is a
        comma-seperated file with the following columns:

        ,bias,rkm,branch,is_lmw,std,mae,max13,last25

        with for each station:

        - bias = bias, mean error
        - rkm = river kilometer of the station
        - branch = name of 1D branch on which the station lies
        - is_lmw = if "LMW" is in the name of station, True.
        - std = standard deviation of the rror
        - mae = mean absolute error of the error

        """

        self.statistics = self._compute_statistics()

        statfile = self.output_path.joinpath(file_path).with_suffix(".csv")
        sumfile = self.output_path.joinpath(file_path + "_summary").with_suffix(".csv")

        # all statistics
        self.statistics.to_csv(statfile)
        self.set_logger_message(f"statistics written to {statfile}")

        # summary of statistics
        s = self.statistics
        with open(sumfile, "w") as f:
            for branch in s.branch.unique():
                bbias = s.bias[s.branch == branch].mean()
                bstd = s["std"][s.branch == branch].mean()
                lmw_bias = s.bias[(s.branch == branch) & s.is_lmw].mean()
                lmw_std = s["std"][(s.branch == branch) & s.is_lmw].mean()
                f.write(
                    f"{branch},{bbias:.2f}±({bstd:.2f}), {lmw_bias:.2f}±({lmw_std:.2f})\n"
                )

    def figure_at_station(
        self, station: str, func: str = "time", savefig: bool = True
    ) -> FigureOutput:
        """
        Creates a figure with the timeseries at a single observation station.

        ``` py

        ```

        Parameters:
            station: name of station. use `stations` method to list all station names
            func: use `time` for a timeseries and `qh` for rating curve
            savefig: if True, saves to png. If False, returned FigureOutput



        """

        fig, ax = plt.subplots(1, figsize=(12, 5))
        error_ax = ax.twinx()

        # q/h view
        match func.lower():
            case "qh":
                ax.plot(
                    self._qsteps,
                    self._data_2D_H_digitized[station],
                    "--",
                    linewidth=2,
                    label="2D",
                )
                ax.plot(
                    self._qsteps,
                    self._data_1D_H_digitized[station],
                    "-",
                    linewidth=2,
                    label="1D",
                )
                ax.set_title(f"{station}\nQH-relatie")
                ax.set_title("QH-relatie")
                ax.set_xlabel("Afvoer [m$^3$/s]")
                ax.set_ylabel("Waterstand [m+NAP]")
                error_ax.plot(
                    self._qsteps,
                    self._data_1D_H_digitized[station]
                    - self._data_2D_H_digitized[station],
                    ".",
                    color=self._color_error,
                )
            case "time":
                ax.plot(self.data_2D_H[station], "--", linewidth=2, label="2D")
                ax.plot(self.data_1D_H[station], "-", linewidth=2, label="1D")

                ax.set_ylabel("Waterstand [m+NAP]")
                ax.set_title(f"{station}\nTijdreeks")

                error_ax.plot(
                    self.data_1D_H[station] - self.data_2D_H[station],
                    ".",
                    label="1D-2D",
                    color=self._color_error,
                )

        # statistics
        stats = self._get_statistics(station)

        stats_labels = [
            f"bias={stats['bias']:.2f} m",
            f"std={stats['std']:.2f} m",
            f"MAE={stats['mae']:.2f} m",
        ]
        stats_handles = [mpatches.Patch(color="white")] * len(stats_labels)

        # Style
        fig, lgd = PlotStyles.apply(
            fig=fig,
            style=self._plotstyle,
            use_legend=True,
            extra_labels=[stats_handles, stats_labels],
        )

        self._style_error_axes(error_ax, ylim=[-1, 1])

        fig.tight_layout()

        if savefig:
            fig.savefig(
                self.output_path.joinpath("figures/stations").joinpath(
                    f"{station}.png"
                ),
                bbox_extra_artists=[lgd],
                bbox_inches="tight",
            )
            plt.close()
        else:
            return FigureOutput(fig=fig, axes=ax, legend=lgd)

    def _style_error_axes(
        self, ax, ylim: List[float] = [-0.5, 0.5], ylabel: str = "1D-2D [m]"
    ):
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        ax.spines["right"].set_edgecolor(self._color_error)
        ax.tick_params(axis="y", colors=self._color_error)
        ax.grid(False)

    def _compute_statistics(self) -> pd.DataFrame:
        """
        Computes statistics for the difference between 1D and 2D water levels

        Returns DataFrame with
            columns: rkm, branch, is_lmw, bias, std, mae
            rows: observation stations

        """

        diff = self.data_1D_H - self.data_2D_H
        station_names = diff.columns
        rkms = self._names_to_rkms(station_names)
        branches = self._names_to_branches(station_names)

        stats = pd.DataFrame(data=diff.mean(), columns=["bias"])
        stats["rkm"] = rkms
        stats["branch"] = branches
        stats["is_lmw"] = [
            True if "lmw" in name.lower() else False for name in station_names
        ]

        # stats
        stats["bias"] = diff.mean()
        stats["std"] = diff.std()
        stats["mae"] = diff.abs().mean()

        stats["1D_last3"] = self._apply_stat(self.data_1D_H, stat="last3")
        stats["1D_last25"] = self._apply_stat(self.data_1D_H, stat="last25")
        stats["1D_max3"] = self._apply_stat(self.data_1D_H, stat="max3")
        stats["1D_max13"] = self._apply_stat(self.data_1D_H, stat="max13")

        stats["2D_last3"] = self._apply_stat(self.data_2D_H, stat="last3")
        stats["2D_last25"] = self._apply_stat(self.data_2D_H, stat="last25")
        stats["2D_max3"] = self._apply_stat(self.data_2D_H, stat="max3")
        stats["2D_max13"] = self._apply_stat(self.data_2D_H, stat="max13")

        stats["diff_last3"] = self._apply_stat(diff, stat="last3")
        stats["diff_last25"] = self._apply_stat(diff, stat="last25")
        stats["diff_max3"] = self._apply_stat(diff, stat="max3")
        stats["diff_max13"] = self._apply_stat(diff, stat="max13")

        return stats

    def _get_statistics(self, station):
        if self.statistics is None:
            self.statistics = self._compute_statistics()
        return self.statistics.loc[station]

    def figure_compare_discharge_at_stations(
        self, stations: List[str], title: str = "no_title", savefig: bool = True
    ) -> FigureOutput | None:
        """
        Like `Compare1D2D.figure_at_station`, but compares discharge
        distribution over two stations.

        Example usage:
        ``` py
        Compare1D2().figure_compare_discharge_at_stations(stations=["WL_869.00", "PK_869.00"])
        ```
        Figures are saved to `[Compare1D2D.output_path]/figures/discharge`

        Example output:

        .. figure:: figures_utils/discharge/example.png

            Example output figure

        """
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        ax_error = axs[0].twinx()
        ax_error.set_zorder(
            axs[0].get_zorder() - 1
        )  # default zorder is 0 for ax1 and ax2

        if len(stations) != 2:
            print("error: must define 2 stations")

        linestyles_2d = ["-", "--"]
        for j, station in enumerate(stations):
            if station not in self.stations():
                self.set_logger_message(f"{station} not known", "warning")

            # tijdserie
            axs[0].plot(
                self.data_2D_Q[station],
                label=f"2D, {station.split('_')[0]}",
                linewidth=2,
                linestyle=linestyles_2d[j],
            )
            axs[0].plot(
                self.data_1D_Q[station],
                label=f"1D, {station.split('_')[0]}",
                linewidth=2,
                linestyle="-",
            )

        ax_error.plot(
            self._data_1D_Q[station] - self._data_2D_Q[station],
            ".",
            color="r",
            markersize=5,
            label=f"1D-2D",
        )

        # discharge distribution

        Q2D = self.data_2D_Q[stations]
        Q1D = self.data_1D_Q[stations]
        axs[1].plot(
            Q2D.sum(axis=1),
            (Q2D.iloc[:, 0] / Q2D.sum(axis=1)) * 100,
            linewidth=2,
            linestyle="--",
        )
        axs[1].plot(
            Q1D.sum(axis=1),
            (Q1D.iloc[:, 0] / Q1D.sum(axis=1)) * 100,
            linewidth=2,
            linestyle="-",
        )
        axs[1].plot(
            Q2D.sum(axis=1),
            (Q2D.iloc[:, 1] / Q2D.sum(axis=1)) * 100,
            linewidth=2,
            linestyle="--",
        )
        axs[1].plot(
            Q1D.sum(axis=1),
            (Q1D.iloc[:, 1] / Q1D.sum(axis=1)) * 100,
            linewidth=2,
            linestyle="-",
        )

        # style
        axs[1].set_ylim(0, 100)
        axs[1].set_title("afvoerverdeling")
        axs[1].set_ylabel("percentage t.o.v. totaal")
        axs[1].set_xlabel("afvoer bovenstrooms [m$^3$/s]")
        axs[0].set_ylabel("afvoer [m$^3$/s]")
        axs[0].set_title("tijdserie")

        suptitle = plt.suptitle(title.upper())

        # Style figure
        fig, lgd = PlotStyles.apply(fig=fig, style=self._plotstyle, use_legend=True)
        self._style_error_axes(ax_error, ylim=[-500, 500], ylabel="1D-2D [m$^3$/s]")
        fig.tight_layout()

        if savefig:
            fig.savefig(
                self.output_path.joinpath("figures/discharge").joinpath(f"{title}.png"),
                bbox_extra_artists=[lgd, suptitle],
                bbox_inches="tight",
            )
            plt.close()
        else:
            return FigureOutput(fig=fig, axes=axs, legend=lgd)

    def get_data_along_route_for_time(
        self, data: pd.DataFrame, route: List[str], time_index: int
    ) -> pd.Series:
        stations, rkms, _ = self.get_route(route)

        tmp_data = list()
        for station in stations:
            tmp_data.append(data[station].iloc[time_index])

        return pd.Series(index=rkms, data=tmp_data)

    def get_data_along_route(
        self, data: pd.DataFrame, route: List[str]
    ) -> pd.DataFrame:
        stations, rkms, _ = self.get_route(route)

        tmp_data = list()
        for station in stations:
            tmp_data.append(data[station])

        df = pd.DataFrame(index=rkms, data=tmp_data)

        # drop duplicates
        df = df.drop_duplicates()
        return df

    @staticmethod
    def _sec_to_days(seconds):
        return seconds / (3600 * 24)

    @staticmethod
    def _get_nearest_time(data: pd.DataFrame, date: datetime = None) -> int:
        try:
            return list(data.index < date).index(False)
        except ValueError:
            # False is not list, return last index
            return len(data.index) - 1

    def _time_func(self, route) -> Dict[str, pd.Series | str]:
        first_day = self.data_1D_H.index[0]  # + timedelta(days=delta_days) * 2
        last_day = self.data_1D_H.index[-1]
        number_of_days = (last_day - first_day).days
        delta_days = int(number_of_days / 6)

        moments = [
            first_day + timedelta(days=i) for i in range(0, number_of_days, delta_days)
        ]
        lines = []

        for day in moments:
            h1d = self.get_data_along_route_for_time(
                data=self.data_1D_H,
                route=route,
                time_index=self._get_nearest_time(data=self.data_1D_H, date=day),
            )

            h2d = self.get_data_along_route_for_time(
                data=self.data_2D_H,
                route=route,
                time_index=self._get_nearest_time(data=self.data_2D_H, date=day),
            )

            lines.append({"1D": h1d, "2D": h2d, "label": f"{day:%b-%d}"})

        return lines

    @staticmethod
    def _apply_stat(df, stat: str = "max13"):
        """
        Applies column-wise "last25" or "max13" on 1D and 2D data
        """
        columns = df.columns
        values = []
        for column in columns:
            try:
                af = df[column].iloc[:, 0]
            except pd.errors.IndexingError:
                af = df[column]
            match stat:
                case "max3":
                    values.append(af.nlargest(3).mean())
                case "max13":
                    values.append(af.nlargest(13).mean())
                case "last3":
                    values.append(af[-3:].mean())
                case "last25":
                    values.append(af[-25:].mean())
        return pd.Series(index=columns, data=values)

    def _stat_func(
        self, route: List[str], stat: str = "max13"
    ) -> List[Dict[str, pd.Series | str]]:
        """
        Applies column-wise "last25" or "max13" on 1D and 2D data
        """
        max13_1d = self._apply_stat(
            self.get_data_along_route(self.data_1D_H, route=route).T, stat=stat
        )
        max13_2d = self._apply_stat(
            self.get_data_along_route(self.data_2D_H, route=route).T, stat=stat
        )

        return [{"1D": max13_1d, "2D": max13_2d, "label": stat}]

    def _lmw_func(self, station_names, station_locs):
        st_names = []
        st_locs = []
        prev_loc = -9999
        for name, loc in zip(station_names, station_locs):
            if "lmw" not in name.lower():
                continue
            if abs(prev_loc - loc) < 5:
                self.set_logger_message(
                    f"skipped labelling {name} because too close to previous station",
                    "warning",
                )
                continue
            st_names.append(name.split("_")[-1])
            st_locs.append(loc)
            prev_loc = loc

        return st_names, st_locs

    def figure_longitudinal_time(self, route: List[str]) -> None:
        warnings.warn(
            'Method figure_longitudinal_time will be removed in the future. Use figure_longitudinal(route, stat="time") instead ',
            category=DeprecationWarning,
        )

        self.figure_longitudinal(route, stat="time")

    def figure_longitudinal(
        self,
        route: List[str],
        stat: str = "time",
        savefig: bool = True,
        label: str = "",
        add_to_fig: FigureOutput | None = None,
    ) -> FigureOutput | None:
        """
        Creates a figure along a `route`. Content of figure depends
        on `stat`. Figures are saved to `[Compare1D2D.output_path]/figures/longitudinal`

        Example output:

        ![title](../figures/test_results/compare1d2d/BR-PK-IJ.png)

        Parameters:
            route: List of branches (e.g. ['NK', 'LK'])
            stat: what type of longitudinal plot to make. Options are:
                - time
                - last25
                - max13
            savefig: if true, figure is saved to png file. If false, `FigureOutput`
                     returned, which is input for `add_to_fig`
            add_to_fig: if `FigureOutput` is provided, adds content to figure.
        """
        # Get route and stations along route
        routename = "-".join(route)

        # Make configurable in the future
        labelfunc = self._lmw_func

        # TIME FUNCTION plot line every delta_days days
        match stat:
            case "time":
                lines = self._time_func(route=route)
            case y if y in ["last3", "last25", "max3", "max13"]:
                lines = self._stat_func(stat=y, route=route)
            case _:
                raise KeyError(f"{stat} is unknown statistics")

        # Get figure object
        if add_to_fig is None:
            fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        else:
            fig = add_to_fig.fig
            axs = add_to_fig.axes

        # Filtering which stations to plot
        if add_to_fig is None:
            station_names, station_locs, _ = self.get_route(route)
            st_names, st_locs = labelfunc(station_names, station_locs)
            h1d = self.get_data_along_route(data=self.data_1D_H, route=route)
            for st_name, st_loc in zip(st_names, st_locs):
                for ax in axs:
                    ax.axvline(x=st_loc, linestyle="--")

                axs[0].text(
                    st_loc,
                    h1d.min().min(),
                    st_name,
                    fontsize=12,
                    rotation=90,
                    horizontalalignment="left",
                    verticalalignment="bottom",
                )

        for line in lines:
            axs[0].plot(line.get("1D"), label=f"{label} {line.get('label')}")

            axs[0].set_ylabel("Waterstand [m+NAP]")
            routestr = "-".join(route)

            axs[0].set_title(f"route: {routestr}")

            axs[1].plot(line.get("1D") - line.get("2D"))
            axs[1].set_ylabel("Verschil 1D-2D [m]")

            for ax in axs:
                ax.set_xlabel("Rivierkilometers")
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))

        axs[1].set_ylim(-1, 1)
        fig, lgd = PlotStyles.apply(fig=fig, style=self._plotstyle, use_legend=True)

        if savefig:
            plt.tight_layout()
            fig.savefig(
                self.output_path.joinpath(f"figures/longitudinal/{routename}.png"),
                bbox_extra_artists=[lgd],
                bbox_inches="tight",
            )
            plt.close()

        else:
            return FigureOutput(fig=fig, axes=axs, legend=lgd)

    def figure_longitudinal_rating_curve(self, route: List[str]) -> None:
        """
        Create a figure along a route with lines at various dicharges.
        To to this, rating curves are generated at each point by digitizing
        the model output.

        Figures are saved to `[Compare1D2D.output_path]/figures/longitudinal`

        Example output:

        .. figure:: figures_utils/longitudinal/example_rating_curve.png

            example output figure

        """
        routename = "-".join(route)
        _, _, lmw_stations = self.get_route(route)

        h1d = self.get_data_along_route(data=self._data_1D_H_digitized, route=route)
        h2d = self.get_data_along_route(data=self._data_2D_H_digitized, route=route)

        discharge_steps = list(self._iter_discharge_steps(h1d.T, n=8))
        if len(discharge_steps) < 1:
            self.set_logger_message(
                "There is too little data to plot a QH relationship", "error"
            )
            return

        # Plot LMW station locations
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        prevloc = -9999
        for lmw in lmw_stations:
            if lmw is None:
                continue
            newloc = max(lmw[1], prevloc + 3)
            prevloc = lmw[1]
            for ax in axs:
                ax.axvline(x=lmw[1], linestyle="--", color="#7a8ca0")
            axs[0].text(
                newloc,
                h1d[discharge_steps[0]].min(),
                lmw[0],
                fontsize=12,
                rotation=90,
                horizontalalignment="right",
                verticalalignment="bottom",
            )

        # Plot betrekkingslijnen
        texty_previous = -999
        for discharge in discharge_steps:
            axs[0].plot(h1d[discharge], label=f"{discharge:.0f} m$^3$/s")
            axs[0].set_ylabel("waterstand [m+nap]")
            routestr = "-".join(route)

            axs[0].set_title(f"Betrekkingslijnen\n{routestr}")

            axs[1].plot(h1d[discharge] - h2d[discharge])
            axs[1].set_ylabel("Verschil 1D-2D [m]")

            for ax in axs:
                ax.set_xlabel("rivierkilometers")
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))

        # style figure
        axs[1].set_ylim(-1, 1)
        fig, lgd = PlotStyles.apply(fig, style=self._plotstyle, use_legend=True)
        plt.tight_layout()
        fig.savefig(
            self.output_path.joinpath(
                f"figures/longitudinal/{routename}_rating_curve.png",
            ),
            bbox_extra_artists=[lgd],
            bbox_inches="tight",
        )
        plt.close()

    def _iter_discharge_steps(self, data: pd.DataFrame, n: int = 5) -> List[float]:
        """
        Choose discharge steps based on increase in water level downstream
        """
        station = data[data.columns[-1]]

        wl_range = station.max() - station[station.index > 0].min()

        stepsize = wl_range / (n + 1)
        q_at_t_previous = 0
        for i in range(1, n + 1):
            t = station[station.index > 0].min() + i * stepsize
            q_at_t = (station.dropna() < t).idxmin()
            if q_at_t == q_at_t_previous:
                continue
            q_at_t_previous = q_at_t
            yield q_at_t

    def heatmap_time(self, route: List[str]) -> None:
        """
        Create a 2D heatmap along a route. The horizontal axis uses
        timemarks to match the 1D and 2D models

        Figures are saved to `[Compare1D2D.output_path]/figures/heatmap`

        Example output:

        .. figure:: figures_utils/heatmaps/example_time_series.png

            example output figure

        """

        routename = "-".join(route)
        _, _, lmw_stations = self.get_route(route)
        data = self._data_1D_H - self._data_2D_H
        routedata = self.get_data_along_route(data.dropna(how="all"), route)

        fig, ax = plt.subplots(1, figsize=(12, 7))
        im = ax.pcolormesh(
            routedata.columns,
            routedata.index,
            routedata,
            cmap="Spectral_r",
            vmin=-1,
            vmax=1,
        )
        for lmw in lmw_stations:
            if lmw is None:
                continue
            ax.plot(
                routedata.columns, [lmw[1]] * len(routedata.columns), "--k", linewidth=1
            )
            ax.text(routedata.columns[0], lmw[1], lmw[0], fontsize=12)

        ax.set_ylabel("rivierkilometer")
        ax.set_title(f"{routename}\nheatmap Verschillen in waterstand 1D-2D")

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("waterstandsverschil [m+nap]".upper(), rotation=270, labelpad=15)
        PlotStyles.apply(fig, style=self._plotstyle, use_legend=False)
        fig.tight_layout()
        fig.savefig(
            self.output_path.joinpath(f"figures/heatmaps/{routename}_timeseries.png")
        )
        plt.close()

    def heatmap_rating_curve(self, route: List[str]) -> None:
        """
        Create a 2D heatmap along a route. The horizontal axis uses
        the digitized rating curves to match the two models

        Figures are saved to `[Compare1D2D.output_path]/figures/heatmap`

        Example output:

        .. figure:: figures_utils/heatmaps/example_rating_curve.png

            example output figure

        """

        routename = "-".join(route)
        _, _, lmw_stations = self.get_route(route)
        data = self._data_1D_H_digitized - self._data_2D_H_digitized

        routedata = self.get_data_along_route(data.dropna(how="all"), route)

        fig, ax = plt.subplots(1, figsize=(12, 7))
        im = ax.pcolormesh(
            routedata.columns,
            routedata.index,
            routedata,
            cmap="Spectral_r",
            vmin=-1,
            vmax=1,
        )
        for lmw in lmw_stations:
            if lmw is None:
                continue
            ax.plot(
                routedata.columns, [lmw[1]] * len(routedata.columns), "--k", linewidth=1
            )
            ax.text(routedata.columns[0], lmw[1], lmw[0], fontsize=12)

        ax.set_ylabel("rivierkilometer")
        ax.set_title(f"{routename}\nheatmap Verschillen in waterstand 1D-2D")

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("waterstandsverschil [m+nap]".upper(), rotation=270, labelpad=15)
        PlotStyles.apply(fig, style=self._plotstyle, use_legend=False)
        fig.tight_layout()
        fig.savefig(
            self.output_path.joinpath(f"figures/heatmaps/{routename}_rating_curve.png")
        )
        plt.close()

    @staticmethod
    def _catch_e(
        func: Callable, exception: Union[Exception, Tuple[Exception]], *args, **kwargs
    ):
        """catch exception in function call. useful for list comprehensions"""
        try:
            return func(*args, **kwargs)
        except exception as e:
            return np.nan

